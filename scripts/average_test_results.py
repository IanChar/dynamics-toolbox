"""
Average the test results written and report.

Author: Ian Char
Date: Feburary 6, 2023
"""
import argparse
from collections import defaultdict
import os

import numpy as np

###########################################################################
# %% The arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)  # Parent directory of seeds.
args = parser.parse_args()

###########################################################################
# %% Load in the test results.
###########################################################################
results = defaultdict(list)
for seed in os.listdir(args.dir):
    if 'pkl' not in seed:
        files = os.listdir(os.path.join(args.dir, seed))
        if 'test_results.txt' in files:
            with open(os.path.join(args.dir, seed, 'test_results.txt')) as f:
                lines = f.readlines()
            for line in lines:
                k, val = line.split(': ')
                results[k].append(float(val))

###########################################################################
# %% Display the averages.
###########################################################################
for k, d in results.items():
    mean = np.mean(d)
    err = np.std(d) / np.sqrt(len(d))
    print(f'{k}: {mean} +- {err}')
