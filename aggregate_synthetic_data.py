#!/usr/bin/env python
# coding: utf-8

import argparse
from os import listdir
from os.path import isfile, join
import typing
from typing import List
import pandas as pd

def main(args):
    data_path = args[0]
    tsv_files = [f for f in listdir(data_path) if f.endswith(".tsv")]

    all_dfs = [pd.read_csv(join(data_path, f), sep="\t") for f in tsv_files]

    merged_df = all_dfs[0]

    for df in all_dfs[1:]:
        if df.shape[1] != 100:
            if df.shape[1] == 101:
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