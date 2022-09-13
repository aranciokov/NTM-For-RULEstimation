import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# tss: time series grouped by engID (each group = one time series)
# proposed in [Elsheikh et al., 2019]
def new_RUL_label_gen(ts, max_rul):
    # ts: (seq, inp_size)
    # lifespan: (seq, )
    sens_data = ts.shape[1]
    seq_len = len(ts)
    w = 2
    lifespan = [seq_len - i - (w - 1) - 1 for i in range(seq_len - (w - 1))]
    mlfs = np.asarray(lifespan * sens_data)\
        .reshape(sens_data, seq_len - (w - 1))\
        .transpose(1, 0)
    # (1) moving average smoothing column-wise
    tsr = ts.rolling(window=w, axis=0)
    tsr_mean = tsr.mean()
    tsr_mean = tsr_mean[w-1:]
    # ^ contains NaN at the beginning (due to the window), dropping

    # (2) scaling down and shifting
    s = MinMaxScaler()
    f = s.fit_transform(tsr_mean)

    # (*) upward trends should be reversed
    for ci in range(sens_data):
        column = f[:, ci]
        p, _ = np.polyfit(list(range(len(column))), column, deg=1)
        # p > 0 => upward trend, p <= 0 downward trend
        if p > 0:
            f[:, ci] = np.flip(column, axis=0)

    # (3) scaling up and truncating
    f = f * mlfs
    f = np.clip(f, 0, max_rul)

    # (4) choosing the best/lowest RUL
    f = np.amin(f, axis=1)

    # (*) left-extending the truncated part to cover the initially removed NaN's
    add = [max_rul] * (w - 1)
    f = np.concatenate((add, f), axis=0)
    # final result shape, f: (seq, )
    return f.tolist()


def standard_RUL_label_gen(c, max_rul):
    if c <= max_rul:
        new_ruls = [c - x for x in range(c)]
    else:
        new_ruls = ([max_rul for _ in range(c - max_rul)]
                    + [max_rul - x for x in range(max_rul)])
    return new_ruls
