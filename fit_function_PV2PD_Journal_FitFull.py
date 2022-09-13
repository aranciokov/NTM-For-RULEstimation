import torch
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
#import seaborn as sns
import matplotlib.pylab as pyl
import score_fun
import torch.nn.functional as F

TRAINING = 'training'
VALIDATING = 'validating'
TESTING = 'testing'


import time
def fit(epoch, model, data_loader, phase, num_epochs, windows_size,
        is_cuda, verbose, optimizer, criterion, batch_size, return_mae=False, only_last=False):
    running_loss = 0.0
    running_loss_mae = 0.0
    running_correct = 0
    score = 0.0
    preds = []
    targets = []

    if phase == TRAINING:
        model = model.train()
    else:
        model = model.eval()

    start = time.time()
    for batch_idx, (data, target) in enumerate(data_loader):
        end = time.time()
        #print("batch loaded in " + str(end - start))

        # --- 
        needed_padding = False
        if (phase == VALIDATING or phase == TESTING) and data.shape[0] != batch_size:
            needed_padding = True
            prev_batch_size = data.shape[0]
            prev_shape = data.shape
            data = F.pad(data, (0, 0, 0, 0, 0, batch_size - data.shape[0]), 'constant', 0)
            new_shape = data.shape
            #print("WARNING: padding data, from", prev_shape, "to", new_shape)

        start_cuda = time.time()
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        end_cuda = time.time()
        #print("batch loaded in GPU in " + str(end_cuda - start_cuda))

        start_model = time.time()
        if isinstance(model, torch.nn.LSTM):
            output, (hn, cn) = model(data.float())
            #print(data.shape, output.shape)
        else:
            output = model(data.float(), data.shape[1])
        #print(f"shapes: data {data.shape}, target {target.shape}, output {output.shape}")
        end_model = time.time()
        #print("model forward in " + str(end_model - start_model))
        # output: (batch_size, seq_len, output_size)
        output = output.squeeze(-1)

        # ---
        if (phase == VALIDATING or phase == TESTING) and needed_padding:
            prev_shape = output.shape
            output = output[:prev_batch_size]
            new_shape = output.shape
            #print("WARNING: removing padding from predicted values: from", prev_shape, "-->", new_shape)

        if criterion is not None:
            loss = criterion(output, target.float())
        # --> running_loss = ( running_loss * batch_idx * batch_size + new_loss * batch_size ) / ( (batch_idx + 1) * batch_size )
        # -->              = batch_size * (running_loss * batch_idx + new_loss) / ( (batch_idx + 1) * batch_size )
        # -->              = (running_loss * batch_idx + new_loss) / (batch_idx + 1)
        start_mse = time.time()
        if only_last:
            running_loss = (running_loss * batch_idx + F.mse_loss(output[:, -1], target[:, -1])) / (batch_idx + 1)
        else:
            running_loss = (running_loss * batch_idx + F.mse_loss(output, target)) / (batch_idx + 1)
        if return_mae:
            from sklearn.metrics import mean_absolute_error as mae_loss
            if only_last:
                running_loss_mae = (running_loss_mae * batch_idx + mae_loss(output.cpu()[:, -1], target.cpu()[:, -1])) / (batch_idx + 1)
            else:
                running_loss_mae = (running_loss_mae * batch_idx + mae_loss(output.cpu(), target.cpu())) / (batch_idx + 1)
        end_mse = time.time()
        start_sf = time.time()
        score += score_fun.score_fun_full(output, target)
        end_sf = time.time()
        #print("MSE computation in " + str(end_mse - start_mse))
        #print("Score computation in " + str(end_sf - start_sf))

        if phase == TRAINING:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        start = time.time()

    loss = running_loss
    rmse = sqrt(loss)
    if return_mae:
        mae = running_loss_mae

    if verbose:
        print('Epoch [{}/{}]: {} loss is {:.4f}, RMSE is {:.4f}, score is {:.4f}'
              .format(epoch + 1, num_epochs, phase, loss, rmse, score))

    if not return_mae:
        return loss, score, rmse
    else:
        return loss, score, rmse, mae
