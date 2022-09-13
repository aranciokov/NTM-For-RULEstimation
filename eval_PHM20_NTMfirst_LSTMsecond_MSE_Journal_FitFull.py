from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import argparse

from modelNTM_Journal import Net as NTM
from modelLSTM_Journal import Net as LSTM
from modelLSTMX2_Journal import Net as LSTMx2

from phm20dataset import PHM20Dataset

from fit_function_PV2PD_Journal_FitFull import fit as fit_HDNN_NTM

import time
from logger_comment import get_logdir, get_comment

def find_best_val_model(folder, run_i):
    from pathlib import Path
    all_models = sorted(Path(os.path.join(folder, "models")).iterdir(), key=os.path.getmtime)
    # run 0 ... run 9
    all_models = [str(m) for m in all_models]
    all_models_run_i = [m for m in all_models if "run_{}".format(run_i) in m]
    return all_models_run_i[-1]
    best_val = 1e18
    best_val_score = 1e22
    best_model = "null"
    for m in all_models_run_i:
        _val = float(m.split("_")[m.split("_").index("valrmse")+1])
        _val_score = float(m.split("_")[m.split("_").index("valscore")+1])
        if _val < best_val or _val_score < best_val_score:
            best_val = min(_val, best_val)
            best_val_score = min(_val_score, best_val_score)
            best_model = m
    return os.path.join(folder, "models", best_model)

def collate_fn(data):
    targets = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x["target"]) for x in data], batch_first=True)
    sensors = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x["sensor_data"]) for x in data], batch_first=True)
    #print(targets.shape, sensors.shape)
    return sensors, targets

def main(folder, eval_only_last_time_step):
    all_mses = []
    all_maes = []

    def parse_args(list_args):
        _args = {}
        for ln in list_args:
            par, val = ln.split("->")
            par, val = par.strip(), val.strip()
            _args[par] = val

        return _args

    # paths are like: dataset_name/random_string/run_name/models/model.pkl
    args_file = folder + "/args.txt"
    try:
        with open(args_file) as f:
            args = parse_args(f.readlines())
            hs1, hs2, ffs, scaler = int(args["hidden_size1"]), int(args["hidden_size2"]), int(
                args["dec_size"]), args["scaler"]
            netIndex = int(args["net_index"])
            print("{} -> hidden sizes {} and {}, decoder {}, scaler {}, window size {}".format(
                (netIndex, "NTM" if netIndex==0 else ("LSTM" if netIndex==4 else "-")), hs1, hs2, ffs, scaler, int(args["window_size"])
            ))
    except:
        print("err1")
        return

    nets = [NTM, None, None, LSTM, LSTMx2, None]
    fitfuncs = [fit_HDNN_NTM]*6

    batch_sizes = [args["batchsize"]]
    batch_size_test = int(args["batchsize"])
    #print("Batch size: " + str(batch_sizes[0]))

    num_tests = int(args["num_tests"])
    drop_last = True  # per windowed

    # net size
    input_size = 5

    if args["window_size"] != -1:
      windows_size = args["window_size"]
    #print("Window size in use: " + str(windows_size))
    
    if netIndex == 0:
        model = NTM(input_size,
                            hs1, hs2,
                            ff_size=ffs, output_size=1, batch_size=int(args["batchsize"]),
                            dropout=0, is_cuda=True, num_layers=1,
                            write_fc_drop=0.1, read_fc_drop=0.1, dec_drop=0.25)
    elif netIndex == 1:
        model = NTM_2(input_size,
                            hs1, hs2,
                            ff_size=ffs, output_size=1, batch_size=int(args["batchsize"]),
                            dropout=0, is_cuda=True, num_layers=1,
                            write_fc_drop=0.1, read_fc_drop=0.1, dec_drop=0.25)
    
    elif netIndex == 4:
        model = LSTMx2(input_size, hs1, hs2, ffs, 1, int(args["batchsize"]), True, 1)
    else:
        return

    for i in range(num_tests):
        try:
            model_file = find_best_val_model(folder, i)
            model.load_state_dict(torch.load(model_file))
        except:
            print("err2")
            return

        model = model.cuda()

        if scaler == "standard":
            scaler = preprocessing.StandardScaler()

        elif scaler == "minmax":
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)  # per HDNN

        start = time.time()
        train_split = float(args["train_split"])
        #print("Training set percentage used as validation: %.2f" % (train_split))
        #print("Train dataset creation ...")
        train_dataset = PHM20Dataset(split="train", split_percent=train_split, window_size=int(args["window_size"]),
                                     scaler=scaler, sliding_step=1, max_rul=int(args["max_rul"]),
                                     all_features=True)
        #print("Test dataset creation ...")
        test_dataset = PHM20Dataset(split="test", window_size=int(args["window_size"]), scaler=scaler, sliding_step=1, max_rul=int(args["max_rul"]),
                                     all_features=True)
        end = time.time()
        #print("Time elapsed: " + str(end - start))

        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                                 collate_fn=collate_fn, drop_last=False, num_workers=int(args["num_workers"]))

        model = model.eval()
        with torch.no_grad():
            _, _, rmse_ntm, mae_ntm = fit_HDNN_NTM(0, model, test_loader, "testing", 0, -1, True, False, None, None, 100,
                                                   return_mae=True, only_last=eval_only_last_time_step)
            all_mses.append(rmse_ntm)
            all_maes.append(mae_ntm)

    import numpy as np
    final_rmse = np.array(all_mses)
    final_mae = np.array(all_maes)
    print("{} -> RMSE {:.2f} +- {:.2f}, MAE {:.2f} +- {:.2f}".format(
        (netIndex, "NTM" if netIndex==0 else ("LSTM" if netIndex==4 else "-")), final_rmse.mean(), final_rmse.std(), final_mae.mean(), final_mae.std()
    ))

if __name__ == "__main__":
    """from multiprocessing import set_start_method

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only_last', action="store_true", default=False)
    parser.add_argument('--exp_dir')
    args = parser.parse_args()
    print("defaulting sliding_step=1 for the test dataset")

    main(args.exp_dir, args.eval_only_last)
    
    