from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
import argparse

from modelNTM_Journal import Net as NTM
from modelLSTM_Journal import Net as LSTM
from modelLSTMX2_Journal import Net as LSTMx2

from turbofandataset_windowed_LiHDNN_Journal import TurbofanDataset_W as TurbofanDataset

from fit_function_HDNN_Mono_Journal_FitFull import fit as fit_HDNN_NTM

import time
from logger_comment import get_comment, get_logdir

def collate_fn(data):
    original_lengths = list(map(lambda x: len(x[0]), data))
    original_targets = list(map(lambda x: x[1], data))
    original_idx = [b for (b, _) in sorted(enumerate(original_lengths), key=lambda i: i[1])]
    original_idx.reverse()

    # (1)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    target = list(map(lambda x: x[1], data))
    lengths = list(map(lambda x: len(x[0]), data))
    data = list(map(lambda x: x[0], data))  # lista di tensori (seq_len, 21)

    # (2)
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True)

    # (3)
    data = torch.nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True)
    target = torch.nn.utils.rnn.pack_padded_sequence(target, lengths, batch_first=True)
    return data, target, original_idx, original_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='FD001')
    parser.add_argument('--batchsize', default=100, type=int)
    parser.add_argument('--max_rul', default=125, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--use_decaylr', default='y')
    parser.add_argument('--scaler', default="minmax")
    parser.add_argument('--train_split', default=0.8, type=float)
    parser.add_argument('--dropout_decoder', default='y')
    parser.add_argument('--dropout_write_fcs', default='y')
    parser.add_argument('--dropout_read_fcs', default='y')
    parser.add_argument('--dropout_decoder_value', default=0.25, type=float)
    parser.add_argument('--dropout_write_fcs_value', default=0.10, type=float)
    parser.add_argument('--dropout_read_fcs_value', default=0.10, type=float)
    parser.add_argument('--net_index', default=0, type=int)
    parser.add_argument('--use_momentum_weightdecay', default='y')
    parser.add_argument('--hidden_size1', default=32, type=int)
    parser.add_argument('--hidden_size2', default=64, type=int)
    parser.add_argument('--dec_size', default=8, type=int)
    parser.add_argument('--window_size', default=200, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_tests', default=10, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    args = parser.parse_args()

    scenario = args.scenario
    learning_rates = [args.lr]  # [1e-3]  # [4e-2, 1e-2, 5e-3, 1e-3]
    print("LR: " + str(learning_rates[0]))
    momentums = [0.6]  # [0, 0.3, 0.6, 0.9]
    num_epochs = args.max_epochs  # 300
    print("Max epochs: %d" % (num_epochs))
    max_rul = args.max_rul  # 130
    print("Max RUL: %d" % (max_rul))
    dropouts = [0.25]  # [0.2, 0.25, 0.3, 0.35]  # [0.15, 0.3]
    num_layers = 1
    
    decayLR = True if args.use_decaylr == 'y' else False
    decay_gamma = 0.6
    decay_step_size = 15
    print("Decay LR: " + str(decayLR))
    if decayLR:
        print("Decay LR step size of %d and gamma of %.2f" % (decay_step_size, decay_gamma))

    drop_zero_var_cols = True
    verbose = True
    # features_O = True if operating settings are to be included in the features
    features_O = False
    # features_H = True if the 6 operating conditions need to be onehot-encoded as part of the features
    features_H = True if scenario in ["FD002", "FD004"] else False

    netIndex = int(args.net_index)
    if netIndex == 0:
        print("=== Importing modelNTM_Journal ===")
        net_name = "NTM1"
    elif netIndex == 1:
        print("=== Importing modelNTM_Journal_2 ===")
        net_name = "NTM2"
    elif netIndex == 2:
        print("=== Importing modelNTM_Journal_3 ===")
        net_name = "NTM3"
    elif netIndex == 3:
        print("=== Importing modelLSTM_Journal ===")
        net_name = "LSTM"
    elif netIndex == 4:
        print("=== Importing modelLSTMx2_Journal ===")
        net_name = "LSTMx2"

    nets = [NTM, None, None, LSTM, LSTMx2, None]
    fitfuncs = [fit_HDNN_NTM]*6

    batch_sizes = [args.batchsize]
    batch_size_test = args.batchsize
    print("Batch size: " + str(batch_sizes[0]))

    num_tests = args.num_tests
    drop_last = True
    n_cluster = 6 if features_H else 1

    # features
    sensor_data_feat = 21
    operating_settings = 3
    operating_conditions = 6

    # net size
    adeq_d0v = (-7 if drop_zero_var_cols else 0)
    input_size = sensor_data_feat +\
                 (adeq_d0v) +\
                 (operating_settings if features_O else 0) +\
                 (operating_conditions if features_H else 0)
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    ff_size = args.dec_size
    print("Hidden sizes: %d and %d" % (hidden_size1, hidden_size2))
    print("FC hidden size: %d" % (ff_size))

    dropout_write_fcs_value = args.dropout_write_fcs_value
    if args.dropout_write_fcs == 'y':
        print("Dropout after write FCs with value: " + str(args.dropout_write_fcs_value))
    else:
        print("Not using dropout after write FCs")
        dropout_write_fcs_value = 0.0

    dropout_read_fcs_value = args.dropout_read_fcs_value
    if args.dropout_read_fcs == 'y':
        print("Dropout after read FCs with value: " + str(args.dropout_read_fcs_value))
    else:
        print("Not using dropout after read FCs")
        dropout_read_fcs_value = 0.0

    dropout_decoder_value = args.dropout_decoder_value
    if args.dropout_decoder == 'y':
        print("Dropout in decoder with value: " + str(args.dropout_decoder_value))
    else:
        print("Not using dropout in decoder")
        dropout_decoder_value = 0.0
    
    windows_size = args.window_size
    print("Window size in use: " + str(windows_size))

    ## optimizer parameters
    weight_decay = args.weight_decay  # L2 penalty

    TRAINING = 'training'
    VALIDATING = 'validating'
    TESTING = 'testing'

    if args.scaler == "standard" or args.scaler == "z":
        scaler = preprocessing.StandardScaler()
        scalers = [scaler]
        if scenario in ["FD002", "FD004"]:
            scaler2 = preprocessing.StandardScaler()
            scaler3 = preprocessing.StandardScaler()
            scaler4 = preprocessing.StandardScaler()
            scaler5 = preprocessing.StandardScaler()
            scaler6 = preprocessing.StandardScaler()
            scalers = [scaler, scaler2, scaler3, scaler4, scaler5, scaler6]

    elif args.scaler == "minmax":
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)  # per HDNN
        scalers = [scaler]
        if scenario in ["FD002", "FD004"]:
            scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
            scaler3 = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
            scaler4 = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
            scaler5 = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
            scaler6 = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
            scalers = [scaler, scaler2, scaler3, scaler4, scaler5, scaler6]

    print("Using %s scaler" % args.scaler)

    kmeans = KMeans(n_clusters=n_cluster, random_state=0)

    if verbose:
        print("Preparing datasets ...")

    start = time.time()
    train_split = args.train_split
    print("Training set percentage used as validation: %.2f" % (train_split))
    print("Train dataset creation ...")
    train_dataset = TurbofanDataset("CMAPSSData/train_"+scenario+".txt", "",
                                    TRAINING, max_rul, scalers, kmeans, drop_zero_var_cols,
                                    windows_size, features_O, features_H, train_split=train_split)
    print("Val dataset creation ...")
    val_dataset = TurbofanDataset("CMAPSSData/train_"+scenario+".txt", "",
                                    VALIDATING, max_rul, scalers, kmeans, drop_zero_var_cols,
                                    windows_size, features_O, features_H, train_split=train_split)

    # print(scaler.mean_)
    print("Test dataset creation ...")
    test_dataset = TurbofanDataset("CMAPSSData/test_"+scenario+".txt", "CMAPSSData/RUL_"+scenario+".txt",
                                   TESTING, max_rul, scalers, kmeans, drop_zero_var_cols,
                                   windows_size, features_O, features_H)
    end = time.time()
    print("Time elapsed: " + str(end - start))


    if verbose:
        print("Checking if CUDA is available ...")
    is_cuda = torch.cuda.is_available()
    device_id = -1
    if is_cuda:
        device_id = torch.cuda.current_device()
        if verbose:
            print("Using GPU" + torch.cuda.get_device_name(device_id))
    else:
        if verbose:
            print("No CUDA found")

    exp_log_dir = get_logdir(args, "CMAPSS", get_comment(args, "CMAPSS", net_name))

    for dropout in dropouts:
        for momentum in momentums:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=collate_fn, drop_last=False,
                                              pin_memory=True, num_workers=args.num_workers)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn, drop_last=False, num_workers=args.num_workers)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                                             collate_fn=collate_fn, drop_last=False, num_workers=args.num_workers)

                    if verbose:
                        print("Creating network model ...")

                    best_validation_RMSE_measurements = []  # store best VAL values per num_tests
                    best_validation_Score_measurements = []  # store best VAL values per num_tests
                    best_val_test_RMSE_measurements = []  # store the TEST value based on best VAL values per num_tests
                    best_val_test_Score_measurements = []  # store the TEST value based on best VAL values per num_tests
                    for i in range(num_tests):
                        logger = SummaryWriter(log_dir=exp_log_dir, comment=get_comment(args, "CMAPSS", net_name, i))
                        best_model = ""
                        best_model_rmse = 0.0
                        best_model_score = 0.0
                        best_model_TEST_rmse = 0.0
                        best_model_TEST_score = 0.0
                        train_time = time.time()

                        minimum_rmse = 1000.0
                        minimum_score = 1000000.0

                        min_TEST_rmse = minimum_rmse
                        min_TEST_score = minimum_score

                        net = nets[netIndex]
                        fit = fitfuncs[netIndex]
                        if netIndex < 3:
                            model = net(input_size,
                                        hidden_size1, hidden_size2,
                                        ff_size=ff_size, output_size=1, batch_size=batch_size,
                                        dropout=dropout, is_cuda=is_cuda, num_layers=num_layers,
                                        write_fc_drop=dropout_write_fcs_value, read_fc_drop=dropout_read_fcs_value, dec_drop=dropout_decoder_value)
                        elif netIndex == 3:  # LSTM
                            model = net(input_size, hidden_size1, ff_size, 1, batch_size, is_cuda, num_layers)
                        elif netIndex == 4:  # LSTMx2
                            model = net(input_size, hidden_size1, hidden_size2, ff_size, 1, batch_size, is_cuda, num_layers)

                        if is_cuda:
                            model = model.cuda()
                        if verbose:
                            print(model)

                        if verbose:
                            print("Creating criterion and optimizer ...")
                        criterion = nn.MSELoss()
                        if args.use_momentum_weightdecay == 'y':
                            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
                                                  weight_decay=weight_decay, momentum=momentum)
                            print("Using momentum (%.2f) and weight_decay (%.2f)" % (momentum, weight_decay))
                            print(optimizer)
                        else:
                            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
                            print("Not using momentum or weight_decay")
                            print(optimizer)
                        #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                        if verbose:
                            print("Training starting ...")

                        plot_loss = []
                        if decayLR:
                            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=decay_gamma)
                            ## writer = SummaryWriter()
                        for epoch in range(num_epochs):
                            start_ep = time.time()
                            model = model.train()
                            loss, _, _ = fit(epoch, model, train_loader, TRAINING, num_epochs,
                                             windows_size, is_cuda, verbose, optimizer, criterion, batch_size)

                            model.eval()
                            with torch.no_grad():
                                _, score, rmse = fit(epoch, model, val_loader, VALIDATING, num_epochs,
                                             windows_size, is_cuda, verbose, optimizer, criterion, batch_size)

                            with torch.no_grad():
                                _, test_score, test_rmse = fit(epoch, model, test_loader, TESTING, num_epochs,
                                             windows_size, is_cuda, verbose, optimizer, criterion, batch_size)

                            logger.add_scalar("train/loss", loss.item(), epoch)
                            logger.add_scalar("val/score", score.item(), epoch)
                            logger.add_scalar("val/RMSE", rmse, epoch)
                            logger.add_scalar("test/score", test_score.item(), epoch)
                            logger.add_scalar("test/RMSE", test_rmse, epoch)

                            plot_loss.append(loss.item())

                            if epoch > 5 and (rmse < minimum_rmse or score < minimum_score):
                                model_w_name = "./NTM_MSE/minimum_" + scenario + \
                                           "/best_weights_FitFull_" + scenario + "_" + net_name + "_hs1_" + str(hidden_size1) + "_hs2_" + str(hidden_size2) + "_ff_" + str(ff_size) + \
                                           "_run_" + str(i) + "_valrmse_" + "%.2f" % (rmse) + "_valscore_" + "%.2f" % (score) + \
                                           "_testrmse_" + "%.2f" % (test_rmse) + "_testscore_" + "%.2f" % (test_score) + ".pkl"
                                import os
                                os.makedirs("./NTM_MSE", exist_ok=True)
                                os.makedirs("./NTM_MSE/minimum_" + scenario, exist_ok=True)
                                torch.save(model.state_dict(), model_w_name)
                                print("Saved: " + model_w_name)
                                minimum_rmse = rmse if rmse < minimum_rmse else minimum_rmse
                                minimum_score = score if score < minimum_score else minimum_score

                                best_model = model_w_name
                                best_model_rmse = rmse
                                best_model_score = score
                                best_model_TEST_rmse = test_rmse
                                best_model_TEST_score = test_score

                            if decayLR:
                                ## for name, param in model.named_parameters():
                                ##     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                                scheduler.step()
                                print("New LR: " + str(scheduler.get_lr()))

                            end_ep = time.time()
                            #print("Epoch done in " + str(end_ep - start_ep))

                        best_validation_RMSE_measurements.append(minimum_rmse)
                        best_validation_Score_measurements.append(minimum_score)
                        best_val_test_RMSE_measurements.append(best_model_TEST_rmse)
                        best_val_test_Score_measurements.append(best_model_TEST_score)

                        train_end_time = time.time()
                        train_time = train_end_time - train_time
                        print("Training (run #%d) finished in %.2f seconds" % (i, train_time))
                        print("Testing starting ...")
                        
                    logger.close()

                    best_validation_RMSE_measurements = torch.Tensor(best_validation_RMSE_measurements)
                    print("Best VAL RMSE values over {} tests\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}"
                          .format(num_tests,
                                  best_validation_RMSE_measurements.mean(), best_validation_RMSE_measurements.std(),
                                  best_validation_RMSE_measurements.min(), best_validation_RMSE_measurements.max())
                          .replace(".", ","))
                    best_validation_Score_measurements = torch.Tensor(best_validation_Score_measurements)
                    print("Best VAL SCORE values over {} tests\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}"
                          .format(num_tests,
                                  best_validation_Score_measurements.mean(), best_validation_Score_measurements.std(),
                                  best_validation_Score_measurements.min(), best_validation_Score_measurements.max())
                          .replace(".", ","))
                    best_val_test_RMSE_measurements = torch.Tensor(best_val_test_RMSE_measurements)
                    print("TEST RMSE considering the best validation model over {} tests\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}"
                          .format(num_tests,
                                  best_val_test_RMSE_measurements.mean(), best_val_test_RMSE_measurements.std(),
                                  best_val_test_RMSE_measurements.min(), best_val_test_RMSE_measurements.max())
                          .replace(".", ","))
                    best_val_test_Score_measurements = torch.Tensor(best_val_test_Score_measurements)
                    print("TEST Score considering the best validation model over {} tests\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}"
                          .format(num_tests,
                                  best_val_test_Score_measurements.mean(), best_val_test_Score_measurements.std(),
                                  best_val_test_Score_measurements.min(), best_val_test_Score_measurements.max())
                          .replace(".", ","))


if __name__ == "__main__":
    """from multiprocessing import set_start_method

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass"""

    main()
