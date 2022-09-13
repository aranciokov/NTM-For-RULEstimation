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

def collate_fn(data):
    targets = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x["target"]) for x in data], batch_first=True)
    sensors = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x["sensor_data"]) for x in data], batch_first=True)
    #print(targets.shape, sensors.shape)
    return sensors, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=100, type=int)
    parser.add_argument('--max_rul', default=-1, type=int)
    parser.add_argument('--pwl_runmean', action="store_true", default=False)
    parser.add_argument('--all_features', action="store_true", default=False)
    parser.add_argument('--kmeans_profile', action="store_true", default=False)
    parser.add_argument('--num_kmeans_profile', type=int, default=6)
    parser.add_argument('--rul_shift', default=0, type=int)
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
    parser.add_argument('--dec_sigmoid', action="store_true", default=False)
    parser.add_argument('--bal_split', action="store_true", default=False)
    parser.add_argument('--window_size', default=200, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_tests', default=10, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--stride', default=100, type=int)
    parser.add_argument('--train_pct', default=1.0, type=float)
    args = parser.parse_args()

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

    verbose = True

    netIndex = args.net_index
    if netIndex == 0:
        print("=== Importing modelNTM_Journal ===")
        net_name = "NTM1"
    elif netIndex == 3:
        print("=== Importing modelLSTM_Journal ===")
        net_name = "LSTM"
    elif netIndex == 4:
        print("=== Importing modelLSTMx2_Journal ===")
        net_name = "LSTMx2"
        if args.dec_sigmoid:
            net_name = "LSTMx2 (dec sigm)"
            
    if args.pwl_runmean:
        net_name += " (PwL-RUL)"
    if args.bal_split:
        net_name += " (bal split)"
    if args.train_pct < 1.0:
        net_name += f" ({args.train_pct}trn)"

    nets = [NTM, None, None, LSTM, LSTMx2, None]
    fitfuncs = [fit_HDNN_NTM]*6

    batch_sizes = [args.batchsize]
    batch_size_test = args.batchsize
    print("Batch size: " + str(batch_sizes[0]))

    results_file = f"results_PHM20.csv"
    results_file_mod = "a"
    if not os.path.exists(results_file):
        initial_string = "Net name, Batch size, Max RUL, Max epochs, LR, Use decay LR, Scaler, Train split, " \
                         "Use Dropout decoder, Use Dropout Write FC, Use Dropout Read FC, " \
                         "Dropout decoder value, Dropout Write FC value, Dropout Read FC value, " \
                         "Stride, KMeans profile (-1 or number of clusters), Input size, " \
                         "Hidden size 1, Hidden size 2, Decoder size, Window size, Num Tests, " \
                         "Val RMSE avg, Val RMSE std, Val RMSE min, Val RMSE max, " \
                         "Val Score avg, Val Score std, Val Score min, Val Score max, " \
                         "Test RMSE avg, Test RMSE std, Test RMSE min, Test RMSE max, " \
                         "Test Score avg, Test Score std, Test Score min, Test Score max, " \
                         "\n"
        with open(results_file, "w") as resf:
            resf.write(initial_string)

    num_tests = args.num_tests
    drop_last = True

    # net size
    input_size = 3
    if args.all_features:
        input_size += 2
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
    if args.window_size != -1:
        print("Window size in use: " + str(windows_size))
    use_sliding_windows = False

    ## optimizer parameters
    weight_decay = args.weight_decay  # L2 penalty

    TRAINING = 'training'
    VALIDATING = 'validating'
    TESTING = 'testing'

    if args.scaler == "standard" or args.scaler == "z":
        scaler = preprocessing.StandardScaler()
        if args.kmeans_profile:
            scaler = [preprocessing.StandardScaler()]*args.num_kmeans_profile

    elif args.scaler == "minmax":
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)  # per HDNN
        if args.kmeans_profile:
            scaler = [preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)]*args.num_kmeans_profile

    print("Using %s scaler" % args.scaler)

    if verbose:
        print("Preparing datasets ...")

    start = time.time()
    train_split = args.train_split
    print("Training set percentage used as validation: %.2f" % (train_split))
    print("Train dataset creation ...")
    kmeans_profiler = None
    if args.kmeans_profile:
        kmeans_profiler = KMeans(n_clusters=args.num_kmeans_profile)
    train_dataset = PHM20Dataset(split="train", split_percent=args.train_split, window_size=args.window_size,
                                 scaler=scaler, sliding_step=args.stride, max_rul=args.max_rul, all_features=args.all_features,
                                 compute_kmeans_profile=args.kmeans_profile, kmeans_profiler=kmeans_profiler,
                                 pwl_runningmean=args.pwl_runmean, balanced_split=args.bal_split, train_pct=args.train_pct)
    print("Val dataset creation ...")
    val_dataset = PHM20Dataset(split="val", split_percent=1-args.train_split, window_size=args.window_size,
                               scaler=scaler, sliding_step=args.stride, max_rul=args.max_rul, all_features=args.all_features,
                               compute_kmeans_profile=args.kmeans_profile, kmeans_profiler=kmeans_profiler,
                               pwl_runningmean=args.pwl_runmean, balanced_split=args.bal_split)
    print("Test dataset creation ...")
    test_dataset = PHM20Dataset(split="test", window_size=args.window_size, scaler=scaler, sliding_step=args.stride,
                                max_rul=args.max_rul, all_features=args.all_features,
                                compute_kmeans_profile=args.kmeans_profile, kmeans_profiler=kmeans_profiler,
                                pwl_runningmean=args.pwl_runmean)
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

    exp_log_dir = get_logdir(args, "PHM20", get_comment(args, "PHM20", net_name) + f"RULshift{args.rul_shift}")
    with open(os.path.join(exp_log_dir, "args.txt"), "w") as f:
        f.writelines([f"{k} -> {v}\n" for k, v in args.__dict__.items()])
    os.makedirs(os.path.join(exp_log_dir, "models"), exist_ok=True)

    for dropout in dropouts:
        for momentum in momentums:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=collate_fn, drop_last=drop_last,
                                              num_workers=args.num_workers)
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
                        logger = SummaryWriter(log_dir=exp_log_dir,
                                               comment=get_comment(args, "PHM20", net_name, i) + f"RULshift{args.rul_shift}")
                        best_model = ""
                        best_model_rmse = 0.0
                        best_model_score = 0.0
                        best_model_TEST_rmse = 0.0
                        best_model_TEST_score = 0.0
                        train_time = time.time()

                        min_TEST_rmse = 10e5
                        minimum_rmse = 10e5
                        min_TEST_score = 10e15
                        minimum_score = 10e15

                        # parametri=larghezza vocab (src|tgt) serve al embed layer
                        if netIndex >= 0:
                            net = nets[netIndex]
                            fit = fitfuncs[netIndex]
                            if netIndex < 3:
                                model = net(input_size,
                                            hidden_size1, hidden_size2,
                                            ff_size=ff_size, output_size=1, batch_size=batch_size,
                                            dropout=dropout, is_cuda=is_cuda, num_layers=num_layers,
                                            write_fc_drop=dropout_write_fcs_value, read_fc_drop=dropout_read_fcs_value, dec_drop=dropout_decoder_value)
                            elif netIndex == 3:  # LSTM
                                model = net(input_size, hidden_size1, ff_size, 1, batch_size, is_cuda, num_layers, dec_sigm=args.dec_sigmoid)
                            elif netIndex == 4:  # LSTMx2
                                model = net(input_size, hidden_size1, hidden_size2, ff_size, 1, batch_size, is_cuda, num_layers, dec_sigm=args.dec_sigmoid)

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
                        else:
                            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
                            print("Not using momentum or weight_decay")
                        print(optimizer)

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
                                model_w_name = "best_weights_FitFull_" + "_" + net_name + "_hs1_" + str(hidden_size1) + "_hs2_" + str(hidden_size2) + "_ff_" + str(ff_size) + \
                                           "_run_" + str(i) + "_valrmse_" + "%.2f" % (rmse) + "_valscore_" + "%.2f" % (score) + \
                                           "_testrmse_" + "%.2f" % (test_rmse) + "_testscore_" + "%.2f" % (test_score) + ".pkl"
                                torch.save(model.state_dict(), os.path.join(exp_log_dir, "models", model_w_name))
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
                    
                    with open(results_file, results_file_mod) as resf:
                        resf.write(f"{net_name},{args.batchsize},{args.max_rul},{args.max_epochs},{args.lr},"
                                   f"{args.use_decaylr},{args.scaler},{args.train_split},{args.dropout_decoder},"
                                   f"{args.dropout_write_fcs},{args.dropout_read_fcs},"
                                   f"{'' if not args.dropout_decoder else args.dropout_decoder_value},"
                                   f"{'' if not args.dropout_write_fcs else args.dropout_write_fcs_value},"
                                   f"{'' if not args.dropout_read_fcs else args.dropout_read_fcs_value},"
                                   f"{args.stride},{-1 if not args.kmeans_profile else args.num_kmeans_profile},{input_size},"
                                   f"{args.hidden_size1},{args.hidden_size2},{args.dec_size},{args.window_size},{args.num_tests},"
                                   f"{best_validation_RMSE_measurements.mean():.2f},{best_validation_RMSE_measurements.std():.2f},"
                                   f"{best_validation_RMSE_measurements.min():.2f},{best_validation_RMSE_measurements.max():.2f},"
                                   f"{best_validation_Score_measurements.mean():.2f},{best_validation_Score_measurements.std():.2f},"
                                   f"{best_validation_Score_measurements.min():.2f},{best_validation_Score_measurements.max():.2f},"
                                   f"{best_val_test_RMSE_measurements.mean():.2f},{best_val_test_RMSE_measurements.std():.2f},"
                                   f"{best_val_test_RMSE_measurements.min():.2f},{best_val_test_RMSE_measurements.max():.2f},"
                                   f"{best_val_test_Score_measurements.mean():.2f},{best_val_test_Score_measurements.std():.2f},"
                                   f"{best_val_test_Score_measurements.min():.2f},{best_val_test_Score_measurements.max():.2f},\n"
                                   )


if __name__ == "__main__":
    """from multiprocessing import set_start_method

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass"""

    main()
