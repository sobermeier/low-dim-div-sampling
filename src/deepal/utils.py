import argparse
import ast
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.deepal.settings import DATA_ROOT


def flatten_dict(d):
    """
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def str_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_list(v):
    if isinstance(v, list):
        return v
    return ast.literal_eval(v)


def load_stats_for_run_and_round(out_dir, seeds):
    df_per_seed = dict()
    for seed in seeds:
        dfs_paths = glob.glob(out_dir + '/' + str(seed) + '/*.csv')
        if len(dfs_paths) == 0:
            return None
        dfs = []
        for df_path in dfs_paths:
            iteration = df_path.split("/")[-1].split("_")[0]
            tmp_df = pd.read_csv(df_path)
            tmp_df["iteration"] = int(iteration)
            dfs.append(tmp_df)
        df_per_seed[seed] = pd.concat(dfs[::-1], ignore_index=True)
    return df_per_seed


def load_stats_for_run(out_dir, seeds):
    df_per_seed = dict()
    for seed in seeds:
        dfs_paths = glob.glob(out_dir + '/' + str(seed) + '/*.csv')
        if len(dfs_paths) == 0:
            return None
        dfs = []
        for df_path in dfs_paths:
            dfs.append(pd.read_csv(df_path))
        df_per_seed[seed] = pd.concat(dfs[::-1], ignore_index=True)
    return df_per_seed


def filter_df(filters, df):
    for f in filters:
        df = df[df[f[0]] == f[1]]
    return df
