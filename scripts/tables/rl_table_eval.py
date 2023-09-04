"""
Create an evaluation table from recorded stats.

Author: Ian Char
Date: September 4, 2023
"""
import argparse
from collections import defaultdict
import os

import numpy as np
from tabulate import tabulate

from dynamics_toolbox.utils.storage.rl_stats import read_stats_into_df
from dynamics_toolbox.rl.util.baseline_stats import d4rl_normalize_and_get_baselines

###########################################################################
# %% Constants and set up the plotting.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--parent_path', type=str,
                    default='../AutoCal/logs/aug31_bounding')
parser.add_argument('--stat', type=str, default='Returns/Mean')
parser.add_argument('--legend_map', type=str)
parser.add_argument('--exclude_list', type=str)
parser.add_argument('--exclude_baselines', action='store_true')
parser.add_argument('--average_last', type=float, default=0.1)
parser.add_argument('--table_format', type=str, default='grid')
args = parser.parse_args()
if args.legend_map:
    LEGEND_MAP = {elem.split('=')[0]: elem.split('=')[1]
                  for elem in args.legend_map.split(',')}
else:
    LEGEND_MAP = {}
EXCLUDES = args.exclude_list.split(',') if args.exclude_list else []

###########################################################################
# %% Go through the list and make all of the plots.
###########################################################################
results = defaultdict(dict)
children = os.listdir(args.parent_path)
children.sort()
for child in children:
    child_path = os.path.join(args.parent_path, child)
    for method in os.listdir(child_path):
        if method in EXCLUDES:
            continue
        if method not in LEGEND_MAP:
            LEGEND_MAP[method] = method
        try:
            df = read_stats_into_df(os.path.join(child_path, method))
        except FileNotFoundError:
            print(f'Files not found for {method}. Skipping...')
        df = df.dropna(subset=['Samples', args.stat])
        sample_amounts = df['Samples'].unique()
        sample_cutoff = np.max(sample_amounts) * (1 - args.average_last)
        df_cutoff = df[df['Samples'] >= sample_cutoff]
        avg_results = df_cutoff[args.stat].mean()
        err_results = df_cutoff[args.stat].sem()
        avg_results, err_results, baselines =\
            d4rl_normalize_and_get_baselines(child_path, avg_results, err_results)
        results[child][LEGEND_MAP[method]] = (avg_results, err_results)

###########################################################################
# %% Create the table.
###########################################################################
method_names = list(LEGEND_MAP.values())
headers = ['Environment'] + method_names
rows = []
for k, v in results.items():
    row = [k]
    for mn in method_names:
        row.append(f'{v[mn][0]:0.2f} +- {v[mn][1]:0.2f}')
    rows.append(row)
avg_row = ['Average']
for mn in method_names:
    avg_score = 0
    for v in results.values():
        avg_score += v[mn][0]
    avg_row.append(f'{avg_score / len(results):0.2f}')
rows.append(avg_row)
table = tabulate(rows, headers=headers, tablefmt=args.table_format)
print(table)
