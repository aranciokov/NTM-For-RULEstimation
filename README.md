# Neural Turing Machines for the Remaining Useful Life estimation problem

This repository contains the code accompanying the paper "*Neural Turing Machines for the Remaining Useful Life estimation problem*", accepted for publication in **Computers In Industry**.

#### Python environment
Requirements: python 3, torch 1.3.0 (also tested with 1.7), sklearn 0.24.1, matplotlib 3.3.2, numpy 1.19.5, pandas 1.3.5, tensorboardX 2.2 (should work fine with tensorboard 2.4.1 contained in torch)

#### Training
To launch a training on C-MAPSS dataset, run:

``python rul_estimate_turbofan_NTM_MSE_Journal_FitFull.py``

To launch a training on the PHM Society 2020 Data Challenge dataset, run:

``python rul_estimate_PHM20_NTM_MSE_Journal_FitFull.py``

Several options are available:

- ``--scaler S`` set the scaler to S (either minmax or standard)
- ``--lr LR`` sets the value of the learning rate to LR
- ``--use_decaylr USE`` decays the learning by 0.6 every 15 epochs if USE==y
- ``--train_split TS`` performs train/val split with ratio TS (between 0 and 1)
- ``--dropout_{decoder,write_fcs,read_fcs} D`` adds dropout (with probability P) to the {decoder,write_fcs,read_fcs} if D==y
- ``--dropout_{decoder,write_fcs,read_fcs}_value P`` sets the probability of dropout to P (between 0 and 1) to the {decoder,write_fcs,read_fcs} if D==y
- ``--use_momentum_weightdecay USE`` adds weight decay to the optimization process if USE==y
- ``--weight_decay WD`` sets the value of weight decay to WD
- ``--net_index N`` selects network N to use for the experiments (N=0 for the NTM, N=4 for the LSTM)
- ``--hidden_size1 HS``, ``--hidden_size2 HS``, and ``--dec_size DS`` are used to specify the hyperparameters of the network
- ``--window_size WS`` specifies the length WS of the windows
- ``--num_tests NT`` specifies how many (NT) training runs to perform

Examples of usage:
- LSTM on C-MAPSS FD001: 
```
python rul_estimate_turbofan_NTM_MSE_Journal_FitFull.py \
  --scenario FD001 \
  --batchsize 100 \
  --max_rul 130 \
  --max_epochs 100 \
  --lr 1e-3 \
  --use_decaylr f \
  --scaler standard \
  --net_index 4 \
  --hidden_size1 32 \
  --hidden_size2 64 \
  --dec_size 8 \
  --window_size -1
```

- NTM on PHM Society 2020 Data Challenge: 
```
python rul_estimate_PHM20_NTM_MSE_Journal_FitFull.py \
  --batchsize 100 \
  --max_rul 125 \
  --all_features \
  --max_epochs 200 \
  --lr 5e-3 \
  --use_decaylr f \
  --scaler minmax \
  --net_index 0 \
  --hidden_size1 64 \
  --hidden_size2 64 \
  --dec_size 64 \
  --window_size 210 \
  --num_workers 4 \
  --num_tests 10 \
  --stride 20
```

#### Evaluation
The code used to perform the training on the C-MAPSS dataset also performs the testing (which is reported on screen at the end of the training). 

For the PHM20 dataset, while training the evaluation is performed sparsely in order to reduce the overall training time. To compute the metrics on the testing set for a given experiment (contained in directory D), run:

``python eval_PHM20_NTMfirst_LSTMsecond_MSE_Journal_FitFull.py --eval_only_last --exp_dir D``

For instance, running

``python eval_PHM20_NTMfirst_LSTMsecond_MSE_Journal_FitFull.py --eval_only_last --exp_dir PHM20/TAO5KM6S3Y/PHM20_hs164_hs264_ff64_bs100_ws280_lr0.005_NetNTM1_scalerminmaxRULshift0``

should give you the following output:

``(0, 'NTM') -> hidden sizes 64 and 64, decoder 64, scaler minmax, window size 280
(0, 'NTM') -> RMSE 5.39 +- 1.25, MAE 3.74 +- 0.77``

## Citation
If you use this code as part of any published research, we would really appreciate it if you could cite the following paper:
```text
@article{falcon2022ntm,
  title={Neural Turing Machines for the Remaining Useful Life estimation problem},
  author={Falcon, Alex and D'Agostino, Giovanni and Lanz, Oswald and Brajnik, Giorgio and Tasso, Carlo and Serra, Giuseppe},
  journal={Computers In Industry},
  volume={143},
  year={2022}
}
```

## License

MIT License