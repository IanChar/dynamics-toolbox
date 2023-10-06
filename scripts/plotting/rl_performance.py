"""
Plot RL performance curves.

Author: Ian Char
Date: August 30, 2023
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from dynamics_toolbox.utils.storage.rl_stats import read_stats_into_df
from dynamics_toolbox.rl.util.baseline_stats import d4rl_normalize_and_get_baselines

###########################################################################
# %% Constants and set up the plotting.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='maple_baselines/aug29/walker2d-medium-replay-v0')
parser.add_argument('--env', type=str)
parser.add_argument('--xname', type=str, default='Samples')
parser.add_argument('--yname', type=str, default='Returns/Mean')
parser.add_argument('--ylabel', type=str, default='Average Returns')
parser.add_argument('--diameter', type=int, default=3)
parser.add_argument('--legend_map', type=str)
parser.add_argument('--color_map', type=str)
parser.add_argument('--xlim', type=int)
parser.add_argument('--exclude_list', type=str)
parser.add_argument('--title', type=str)
# parser.add_argument('--plot_individual', action='store_true')
parser.add_argument('--exclude_baselines', action='store_true')
args = parser.parse_args()
plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 16,
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
if args.color_map:
    COLOR_MAP = {elem.split('=')[0]: elem.split('=')[1]
                 for elem in args.color_map.split(',')}
else:
    COLOR_MAP = {}
if args.legend_map:
    LEGEND_MAP = {elem.split('=')[0]: elem.split('=')[1]
                  for elem in args.legend_map.split(',')}
else:
    LEGEND_MAP = {}
EXCLUDES = args.exclude_list.split(',') if args.exclude_list else []
COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'cyan', 'magenta',
          'yellow', 'brown']
LINE_STYLES = ['--', ':', '-', '-.']
color_idx = 0
XLIM = [0, args.xlim]

###########################################################################
# %% Go through the list and make all of the plots.
###########################################################################
# Load in the results.
for method in os.listdir(args.data_path):
    if method in EXCLUDES:
        continue
    if method not in COLOR_MAP:
        COLOR_MAP[method] = COLORS[color_idx % len(COLORS)]
        color_idx += 1
    if method not in LEGEND_MAP:
        LEGEND_MAP[method] = method
    try:
        df = read_stats_into_df(os.path.join(args.data_path, method))
    except FileNotFoundError:
        print(f'Files not found for {method}. Skipping...')
    df = df.dropna(subset=[args.xname, args.yname])
    avg_results = df.groupby([args.xname])[args.yname].mean().to_numpy()
    err_results = df.groupby([args.xname])[args.yname].sem().to_numpy()
    avg_results, err_results, baselines =\
        d4rl_normalize_and_get_baselines(args.data_path, avg_results, err_results)
    xticks = df[args.xname].unique()
    if args.diameter > 0:
        avg_results = np.array([np.mean(avg_results[i-args.diameter:i])
                                for i in range(args.diameter, len(avg_results))])
        err_results = np.array([np.mean(err_results[i-args.diameter:i])
                                for i in range(args.diameter, len(err_results))])
        xticks = xticks[args.diameter:]
    plt.plot(xticks, avg_results, color=COLOR_MAP[method], label=LEGEND_MAP[method],
             alpha=0.7)
    plt.fill_between(
        xticks,
        avg_results - err_results,
        avg_results + err_results,
        color=COLOR_MAP[method],
        alpha=0.2
    )
    ls_idx = 0
    for baseline_name, baseline_stats in baselines.items():
        plt.axhline(baseline_stats[0], ls=LINE_STYLES[ls_idx % len(LINE_STYLES)],
                    color='black', alpha=0.6, label=baseline_name)
        plt.axhspan(
            baseline_stats[0] - baseline_stats[1],
            baseline_stats[0] + baseline_stats[1],
            alpha=0.1,
            color='black',
        )
        ls_idx += 1
plt.xlabel(args.xname)
plt.ylabel(args.ylabel)
plt.xlim(XLIM)
plt.legend()
if args.title:
    plt.title(args.title, fontsize=16)
else:
    plt.title(args.data_path.split('/')[-1], fontsize=16)
plt.show()
