import torch
import pandas as pd
import numpy as np

dust2id = {"ISO 12103-1, A2 Fine Test Dust": 0,
           "ISO 12103-1, A3 Medium Test Dust": 1,
           "ISO 12103-1, A4 Coarse Test Dust": 2}


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class PHM20Dataset(torch.utils.data.Dataset):
    def __init__(self, split, split_percent=0.8, window_size=-1, sliding_step=100, scaler=None, rul_shift=0,
                 max_rul=-1, all_features=False, compute_kmeans_profile=False, kmeans_profiler=None,
                 pwl_runningmean=False, balanced_split=False, train_pct=1.0):
        # train columns: Time(s)  Flow_Rate(ml/m)  Upstream_Pressure(psi)  Downstream_Pressure(psi)
        # test columns: Data_No, Differential_pressure, Flow_rate, Time, Dust_feed, Dust, RUL
        split_name = "Training" if split in ["train", "val"] else "Validation"
        self.all_features = all_features
        self.compute_kmeans_profile = compute_kmeans_profile
        dfs = []
        if split in ["train", "val"]:
            if balanced_split:
                names = [1, 2, 3, 5, 6, 7, 9, 10, 11, 33, 34, 35, 37, 38, 39, 41, 42, 43, 4, 8, 12, 36, 40, 44]
            else:
                names = list(range(1, 13)) + list(range(33, 45))
        else:
            names = list(range(13, 17)) + list(range(45, 49))
        ts_id = 0
        for x in names:
            new_df = pd.read_csv(f"PHM20/{split_name}/Sample{x:02d}.csv")
            new_df["Time_series_ID"] = ts_id
            new_df["Is_broken"] = (new_df["Upstream_Pressure(psi)"] - new_df["Downstream_Pressure(psi)"]) > 20
            first_occurrence = new_df[new_df["Is_broken"] == True].index[0]
            if pwl_runningmean:
                # adapted from https://github.com/zakkum42/phme20-public/blob/main/PHME20_CB_Pipeline.ipynb

                index = np.argmax(running_mean((new_df["Upstream_Pressure(psi)"] - new_df["Downstream_Pressure(psi)"]).values, 5) > 20)
                count = new_df["Time(s)"].count()
                # Linear RUL Assignment
                new_df['Linear'] = None  # in seconds
                pos_rul_list = [c for c in range(index + 1)]
                pos_rul_list.reverse()
                new_df.iloc[:index + 1, new_df.columns.get_loc('Linear')] = pos_rul_list
                neg_rul_list = [-c for c in range(count - index)]
                new_df.iloc[index:, new_df.columns.get_loc('Linear')] = neg_rul_list
                new_df['Linear'] = np.array(new_df['Linear'] / 10.0, dtype=np.float64)  # Sampled at 10 Hz
                #print(type(new_df["Linear"].values[0]), type(new_df["Time(s)"].clip(upper=max_rul).values[0]))
                #exit()

                # Piecewise Linear (PwL) RUL Assignment
                new_df['RUL'] = new_df['Linear'].copy()
                new_df.loc[new_df['RUL'] > max_rul, 'RUL'] = max_rul
            else:
                new_df = new_df.drop(list(range(first_occurrence+1, len(new_df))))
                new_df["RUL"] = max(new_df["Time(s)"]) - new_df["Time(s)"] + rul_shift
                if max_rul > 0:
                    new_df["RUL"] = new_df["RUL"].clip(upper=max_rul)

            if self.all_features:
                new_df["Solid_ratio"] = new_df["RUL"] * 0
                if x in list(range(13, 17)) + list(range(45, 49)):
                    new_df["Solid_ratio"] += 0.475
                elif x in list(range(1, 5)) + list(range(33, 37)):
                    new_df["Solid_ratio"] += 0.4
                elif x in list(range(5, 9)) + list(range(37, 41)):
                    new_df["Solid_ratio"] += 0.425
                elif x in list(range(9, 13)) + list(range(41, 45)):
                    new_df["Solid_ratio"] += 0.45

                new_df["Particle_size"] = new_df["RUL"] * 0
                if x in list(range(1, 13)) + list(range(13, 17)):
                    new_df["Particle_size"] += 0  # "small"
                elif x in list(range(33, 45)) + list(range(45, 49)):
                    new_df["Particle_size"] += 1  # "large"

            ts_id += 1
            dfs.append(new_df)
        df = pd.concat(dfs)

        #print(df)

        if self.all_features:
            feat_names = ["Flow_Rate(ml/m)", "Upstream_Pressure(psi)", "Downstream_Pressure(psi)", "Solid_ratio", "Particle_size"]
        else:
            feat_names = ["Flow_Rate(ml/m)", "Upstream_Pressure(psi)", "Downstream_Pressure(psi)"]
        features = df[feat_names]

        if compute_kmeans_profile:
            if split == "train":
                kmeans_profiler.fit(df[feat_names])
            df["KMeans_prof"] = kmeans_profiler.predict(df[feat_names])
            for i in range(len(scaler)):
                if split == "train":
                    scaler[i] = scaler[i].fit(df[df["KMeans_prof"] == i][feat_names].values)
                features = scaler[i].transform(df[df["KMeans_prof"] == i][feat_names].values)
                df.loc[df["KMeans_prof"] == i, feat_names] = features
            df["KMeans_prof"] = df["KMeans_prof"] / 6
            #print(scaler.scale_)
        else:
            if split == "train":
                scaler = scaler.fit(features.values)
            #print(scaler.scale_)
            features = scaler.transform(features.values)
            df[feat_names] = features

        if split in ["train", "val"]:
            series = df.groupby("Time_series_ID")
            num_series = len(series)
            num_train_series = num_series*split_percent
            if split == "train" and train_pct < 1.0:
                num_train_series = num_series * train_pct
            train_series = df[df["Time_series_ID"] <= num_train_series]  #.groupby("Data_No")
            val_series = df[df["Time_series_ID"] > num_train_series]  #.groupby("Data_No")
            self.data = train_series if split == "train" else val_series

        else:
            self.data = df
        # print(self.data)

        # groups go from 1 to N, not from 0 to N-1
        windowed_data = self.data.groupby("Time_series_ID")
        if window_size > 0:
            tmp = []
            for ts_id in windowed_data.groups:
                ts_data = windowed_data.get_group(ts_id)
                if len(ts_data) > window_size:
                    num_starting_points = len(ts_data) // sliding_step
                    for i in range(num_starting_points):
                        if len(ts_data[i*sliding_step:i*sliding_step+window_size]) < window_size:
                            break
                        tmp.append(ts_data[i*sliding_step:i*sliding_step+window_size])
                else:
                    tmp.append(ts_data)
            self.windowed_data = tmp
        else:
            self.windowed_data = [windowed_data.get_group(g) for g in windowed_data.groups]

        #print(f"# of windows (size={window_size}) for {split} -> {len(self.windowed_data)}")

    def __getitem__(self, item):
        data = self.windowed_data[item]
        dp = data["Flow_Rate(ml/m)"].to_numpy()
        dp = np.reshape(dp, (dp.shape[0], 1))
        fr = data["Upstream_Pressure(psi)"].to_numpy()
        fr = np.reshape(fr, (fr.shape[0], 1))
        df = data["Downstream_Pressure(psi)"].to_numpy()
        df = np.reshape(df, (df.shape[0], 1))
        if self.all_features:
            sr = data["Solid_ratio"].to_numpy()
            sr = np.reshape(sr, (sr.shape[0], 1))
            ps = data["Particle_size"].to_numpy()
            ps = np.reshape(ps, (ps.shape[0], 1))
            sens = np.concatenate([dp, fr, df, sr, ps], -1)
        else:
            sens = np.concatenate([dp, fr, df], -1)

        rul = data["RUL"].to_numpy()

        return {"sensor_data": sens, "target": rul}

    def __len__(self):
        return len(self.windowed_data)