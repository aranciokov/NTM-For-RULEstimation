import os
import string
import random


def get_logdir(args, dataset, comment=None):
    if dataset == "CMAPSS":
        root_path = os.path.join(dataset, args.scenario)
    else:
        root_path = dataset
    root_path = os.path.join(root_path,
                             ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)))
    if comment is not None:
        root_path = os.path.join(root_path, comment)

    os.makedirs(root_path, exist_ok=True)
    return root_path


def get_comment(args, dataset, net_name, i=None):
    com_str = f"{dataset}_hs1{args.hidden_size1}_hs2{args.hidden_size2}_ff{args.dec_size}_bs{args.batchsize}" \
              f"_ws{args.window_size if args.window_size > 0 else 'Full'}" \
              f"_lr{args.lr}" \
              f"_Net{net_name}{'_run{i}' if i is not None else ''}_scaler{args.scaler}"

    return com_str