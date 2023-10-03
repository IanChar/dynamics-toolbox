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
parser.add_argument('--include_list', type=str)
parser.add_argument('--exclude_baselines', action='store_true')
parser.add_argument('--average_last', type=float, default=0.2)
parser.add_argument('--table_format', type=str, default='grid')
parser.add_argument('--flip_rows_and_columns', action='store_true')
parser.add_argument('--normalize', action='store_true')
args = parser.parse_args()
if args.legend_map:
    LEGEND_MAP = {elem.split('=')[0]: elem.split('=')[1]
                  for elem in args.legend_map.split(',')}
else:
    LEGEND_MAP = {}
EXCLUDES = args.exclude_list.split(',') if args.exclude_list else []
INCLUDES = None if args.include_list is None else args.include_list.split(',')
MIN_MAXS = {
    'thruster': (-100.0, 125.0),
    'fusion': (50.0, 90.0),
}

###########################################################################
# %% Go through and collect all of the results.
###########################################################################
results = defaultdict(dict)
children = os.listdir(args.parent_path)
children.sort()
for child in children:
    child_path = os.path.join(args.parent_path, child)
    offset, scale = 0, 1
    if args.normalize:
        for k, v in MIN_MAXS.items():
            if k in child_path:
                offset = v[0]
                scale = (v[1] - v[0]) / 100
                break
    for method in os.listdir(child_path):
        if INCLUDES is not None and method not in INCLUDES:
            continue
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
        err_results = df_cutoff.groupby(['seed'])[args.stat].mean().sem()
        avg_results, err_results, baselines =\
            d4rl_normalize_and_get_baselines(child_path, avg_results, err_results)
        results[child][LEGEND_MAP[method]] = (
            (avg_results - offset) / scale,
            err_results / scale,
        )

###########################################################################
# %% Create the table.
###########################################################################
if args.flip_rows_and_columns:
    env_names = list(results.keys())
    env_names.sort()
    method_names = list(LEGEND_MAP.values())
    method_names.sort()
    headers = ['Method'] + env_names + ['Average']
    rows = []
    for mn in method_names:
        row = [mn]
        avg_val = 0
        for env_name in env_names:
            if mn in results[env_name]:
                row.append(f'{results[env_name][mn][0]:0.2f} +- '
                           f'{results[env_name][mn][1]:0.2f}')
                avg_val += results[env_name][mn][0]
            else:
                row.append('TODO')
        row.append(f'{avg_val / len(env_names):0.2f}')
        rows.append(row)
    table = tabulate(rows, headers=headers, tablefmt=args.table_format)
    print(table)
else:
    method_names = list(LEGEND_MAP.values())
    method_names.sort()
    headers = ['Environment'] + method_names
    rows = []
    for k, v in results.items():
        row = [k]
        for mn in method_names:
            if mn in v:
                row.append(f'{v[mn][0]:0.2f} +- {v[mn][1]:0.2f}')
            else:
                row.append('TODO')
        rows.append(row)
    avg_row = ['Average']
    for mn in method_names:
        avg_score = 0
        for v in results.values():
            if mn in v:
                avg_score += v[mn][0]
            else:
                avg_score += 0
        avg_row.append(f'{avg_score / len(results):0.2f}')
    rows.append(avg_row)
    table = tabulate(rows, headers=headers, tablefmt=args.table_format)
    print(table)
