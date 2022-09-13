from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from rul_gen_19 import new_RUL_label_gen, standard_RUL_label_gen

TRAINING = 'training' 
VALIDATING = 'validating'
TESTING = 'testing'


class TurbofanDataset_W(Dataset):
    def __init__(self, path_to_data, path_to_target, phase, max_rul,
                 scalers, cluster, drop_zero_var_cols, window_size,
                 features_O, features_H, 
                 train_split=0.9):
        self.phase = phase
        self.window_size = window_size
        self.drop_zero_var_cols = drop_zero_var_cols
        column_names = ['engID', 'cycle', 'os1', 'os2',
                        'os3', 's1', 's2', 's3', 's4', 's5',
                        's6', 's7', 's8', 's9', 's10', 's11',
                        's12', 's13', 's14', 's15', 's16', 's17',
                        's18', 's19', 's20', 's21']

        self.data = pd.read_csv(path_to_data, header=None, delim_whitespace=True)
        gp_data = self.data.groupby(0)  # group wrt time series idx
        total_num_ts = len(gp_data.groups)
        self.how_many_to_keep_in_train = int(len(gp_data.groups)*train_split) # n. of how many time series to keep for training
        if phase == TRAINING:
            self.data = self.data[:self.data.index[self.data[0] == self.how_many_to_keep_in_train+1][0]]
            print("Keeping %d (of %d total) time series for training" % (self.how_many_to_keep_in_train, total_num_ts))
        elif phase == VALIDATING:
            self.data = self.data[self.data.index[self.data[0] == self.how_many_to_keep_in_train+1][0]:]  
            self.data = self.data.reset_index(drop=True)
            print("Keeping %d (of %d total) time series for validation" % (total_num_ts - self.how_many_to_keep_in_train, total_num_ts))
        else:  
            print("Found %d time series for %s" % (total_num_ts, phase))
            
        self.data.columns = column_names
        print(self.data.head())

        self.start = 2 if features_O else 5

        new_cols = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
        if features_H:
            op_sets = ['os1', 'os2', 'os3']
            if phase == TRAINING:
                cluster = cluster.fit(self.data[op_sets])
                print("Clusters found:")
                print(cluster.labels_)
                labels = cluster.labels_
            else:
                labels = cluster.predict(self.data[op_sets])

            column_names = column_names + new_cols
            for nc in new_cols:
                self.data[nc] = np.zeros(self.data.__len__())

            self.data.columns = column_names

            # compute the one-hot encoding for the 6 operating conditions
            self.data[new_cols] = pd.get_dummies(labels)

        columns_sensordata = column_names[5:26]
        if drop_zero_var_cols:
            drop_columns = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
            column_names = [cn for cn in column_names if cn not in drop_columns]
            self.data = self.data.drop(columns=drop_columns, axis=1)

        # ctu are the columns over which normalization will be done
        # [2:] if features_O; [5:] otherwise
        # upper limit should be [:26], i.e. sensor_data only
        ctu = column_names[self.start:]
        # z-score normalization is done *only* over sensor data
        # (when doing features_H as well, there are multiple conditions and thus
        #  the normalization is done by first separating these conditions)
        # also note that the scaler is fit only during the training phase

        if phase == TRAINING:
            scalers[0] = scalers[0].fit(self.data[ctu])
            if features_H:
                tmp_data = self.data.groupby(new_cols)
                for cond in tmp_data.groups:
                    i = cond.index(max(cond))
                    # print(tmp_data.get_group(cond).mean(), tmp_data.get_group(cond).std())
                    scalers[i] = scalers[i].fit(tmp_data.get_group(cond)[ctu])
        
        if not features_H:
            self.data[ctu] = scalers[0].transform(self.data[ctu])
        else:
            tmp_data = self.data.groupby(new_cols)
            left = None
            for cond in tmp_data.groups:
                i = cond.index(max(cond))
                group_df = tmp_data.get_group(cond)
                group_df.columns = column_names
                sens = group_df[ctu]
                group_df.loc[:, ctu] = scalers[i].transform(sens)
                group_df.loc[:, ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']] = cond

                if left is not None:
                    v1 = group_df.values
                    v2 = left.values
                    v3 = np.concatenate((v1, v2))
                    # the lexsort is used to make sure that the samples are sort w.r.t.
                    # the first two columns (endID and cycle)
                    ind = np.lexsort((v3[:, 1], v3[:, 0]))
                    v3 = v3[ind]
                    left = pd.DataFrame(v3)
                    del v1
                    del v2
                    del v3
                else:
                    left = group_df

            self.data = left
            self.data.columns = column_names

        # grouped_data contains the time series (one for each engine)
        self.grouped_data = self.data.groupby('engID')

        grouped_target_ruls = []
        engids, steps, ruls = [], [], []

        # at test time the target RULs are given by the dataset itself, whereas for train/val they need to be computed
        #  (the series are otherwise unabelled, since they are known to be run-to-failure)
        if phase == TRAINING or phase == VALIDATING:
            for group in self.grouped_data.groups:
                c = len(self.grouped_data.get_group(group))
                new_ruls = standard_RUL_label_gen(c, max_rul)
                #new_ruls = new_RUL_label_gen(self.grouped_data.get_group(group)[columns_sensordata], max_rul)
                ruls += new_ruls
                steps += addsteps(c)
                engids += [group for _ in range(c)]
                grouped_target_ruls.append(new_ruls)

            self.target = torch.Tensor(ruls)
            self.grouped_target = grouped_target_ruls
        else:
            self.target = pd.read_csv(path_to_target, header=None, delim_whitespace=True)

            for group_idx in self.grouped_data.groups:
                group = self.grouped_data.get_group(group_idx)
                c = len(group)
                tgt = self.target[0][group_idx - 1]
                new_ruls = [tgt for _ in range(c)]
                ruls += new_ruls
                steps += addsteps(c)
                engids += [group_idx for _ in range(c)]
                grouped_target_ruls.append(new_ruls)

            self.grouped_target = grouped_target_ruls

        if self.window_size > 0:
            self.grouped_data = group_chunk(self.grouped_data, self.window_size, self.phase)
            dic = {'engID': engids, 'cycle': steps, 'rul': ruls}
            temp_df = pd.DataFrame(dic)
            self.grouped_target = group_chunk(temp_df.groupby('engID'), self.window_size, self.phase)
        
        print("phase:",phase,"; # of time series:",len(self.grouped_data),"; time window shape:",self.grouped_data.get_group(list(self.grouped_data.groups.keys())[0]).shape,"; # of targets: ",len(self.grouped_target))

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, index):
        if self.window_size > 0:
            if self.phase == TRAINING or self.phase == VALIDATING:
                c = self.grouped_data.groups
                index_r = list(c.keys())[index]
                time_series = self.grouped_data.get_group(index_r).values
                sensor_data = time_series[:, self.start:]
                sensor_data = torch.from_numpy(np.float32(sensor_data))
                target = torch.Tensor(self.grouped_target.get_group(index_r).values)[:, -1]
            else:
                time_series = self.grouped_data.get_group(index + 1).values
                sensor_data = time_series[:, self.start:]
                sensor_data = torch.from_numpy(np.float32(sensor_data))

                target = torch.Tensor(self.grouped_target.get_group(index + 1).values)[:, -1]
        else:
            c = self.grouped_data.groups
            index_r = list(c.keys())[index]
            time_series = self.grouped_data.get_group(index_r).values
            sensor_data = time_series[:, self.start:]
            sensor_data = torch.from_numpy(np.float32(sensor_data))
            target = torch.Tensor(self.grouped_target[index])
        return sensor_data, target


def addsteps(c):
    return list(range(1, c+1))


# used to perform the windowing of the time series
def group_chunk(df, size, phase):
    new_df = []
    new_df_smaller = []
    for ids in df.groups:  # ids = original id of time series
        serie = df.get_group(ids)
        chunks = chunk(serie, size, ids)
        if phase == TRAINING:
            new_df += chunks
        else:
            if len(chunks) == 0:  # e.g. test set FD002 has some shorter series
                cnk = serie
            else:
                cnk = chunks[-1]
            cnk.loc[:, "engID"] = ids
            new_df += [cnk]
    new_df = pd.concat(new_df)
    return new_df.groupby('engID')


def chunk(seq, size, id_serie):
    chunks = []
    for pos in range(0, len(seq) - size + 1, 1):  # e.g. len(seq)=100, size=30 => last window [70..99]
        if len(seq)-pos >= size:
            finestra = seq[pos:pos + size].copy()
            finestra.loc[:, "engID"] = 1000*id_serie + pos
            chunks.append(finestra)
    return chunks

