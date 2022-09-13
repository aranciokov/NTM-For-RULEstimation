from math import exp
def score_fun(preds, gts):
    # preds, gts: [batch]
    h = preds - gts
    s = [ (exp(- h_j/13) -1) if h_j < 0 else (exp(h_j / 10) -1) for h_j in h ]
    score = sum(s)
    return score

def score_fun_PHM08(preds, gts):
    # preds, gts: [batch, seq_len]
    h = preds - gts
    s = []
    for single_seq in h:
        s += [ (exp(- h_j/13) -1) if h_j < 0 else (exp(h_j / 10) -1) for h_j in single_seq ]
    score = sum(s)
    return score


def f(h_j):
  return (exp(- h_j/13) -1) if h_j < 0 else (exp(h_j / 10) -1)

def score_fun_full(preds, gts):
    from torch import flatten
    from numpy import vectorize
    # preds, gts: [batch]
    preds = preds.cpu().detach()
    gts = gts.cpu().detach()
    h = preds - gts
    if len(h.shape) > 1:
      h = flatten(h)
    s = vectorize(f)(h)
    score = sum(s)
    return score
