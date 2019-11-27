#!/usr/bin/env python
# coding: utf-8

import argparse
from os import listdir
from os.path import isfile, join
import pandas as pd

def main(args):
    data_path = args[0]
    # collect the names of all .tsv files, except those with "nonoise" (these have "noise" in the title)
    tsv_files = [f for f in listdir(data_path) if f.endswith(".tsv") and "noise" not in f]
    if len(tsv_files) == 0:
        raise ValueError('No .tsv files found.')
    # load these into a list of pandas DataFrames
    all_dfs = [pd.read_csv(join(data_path, f), sep="\t") for f in tsv_files]
    # we need the number of nodes in the network
    col_size = [df.shape[1] for df in all_dfs]
    num_nodes = max(set(col_size), key=col_size.count) # take the mode as the most probable number of nodes

    merged_df = all_dfs[0]

    for df in all_dfs[1:]:
        if df.shape[1] != num_nodes:
            if df.shape[1] == (num_nodes + 1): # when there is a 'Time' column
                merged_df = merged_df.append(df.iloc[:,1:], sort=True)
            else:
                pass
        else:
            merged_df = merged_df.append(df, sort=True)
    save_path = join(data_path, "merged.csv")
    merged_df.to_csv(save_path)

if __name__ == "main_":
    parser = argparse.ArgumentParser(description='Take the output from SysGenSim and combine the rows into a single .csv file.')
    parser.add_argument("--path", help="path to the folder containing SysGenSim output")
    args = parser.parse_args()
    main(args)