"""
Make GIFs of the pendulum rollout with different models.

Author: Ian Char
Date: January 31, 2023
"""
import argparse
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dynamics_toolbox.utils.storage.model_storage import load_model_from_log_dir
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5


###########################################################################
# %% Load in the arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--save_path', type=str, default='gifs/pendulum/mlp')
parser.add_argument('--data_path', type=str, default='data/pendulum_holdout.hdf5')
parser.add_argument('--path_len', type=int, default=200)
parser.add_argument('--max_paths', type=int, default=3)
parser.add_argument('--num_samples', type=int, default=30)
args = parser.parse_args()
plt.style.use('seaborn')

###########################################################################
# %% Make the GIFs
###########################################################################
model = load_model_from_log_dir(args.model_path)
os.makedirs(args.save_path, exist_ok=True)
os.makedirs('gif_scratch', exist_ok=True)
data = load_from_hdf5(args.data_path)
num_paths = len(data['observations']) // args.path_len
if args.max_paths is not None and args.max_paths < num_paths:
    num_paths = args.max_paths
for path_num in range(num_paths):
    strt = int(path_num * args.path_len)
    model.reset()
    curr_preds = np.array([data['observations'][strt] for _ in range(args.num_samples)])
    for t in tqdm(range(args.path_len), desc=f'Path {path_num + 1}'):
        ypt = [data['observations'][t + strt, 0]]
        xpt = [data['observations'][t + strt, 1]]
        plt.scatter(xpt, ypt, color='black')
        plt.scatter(curr_preds[:, 1], curr_preds[:, 0], alpha=0.2, color='blue')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.savefig(f'gif_scratch/frame{t}.png')
        plt.clf()
        curr_preds = model.predict(np.concatenate([
            curr_preds,
            np.array([data['actions'][strt + t] for _ in range(args.num_samples)]),
        ], axis=1))[0] + curr_preds
    gif_name = os.path.join(args.save_path, f'path_{path_num + 1}.gif')
    with imageio.get_writer(gif_name, mode='I', fps=12) as writer:
        for t in range(args.path_len):
            image = imageio.imread(f'gif_scratch/frame{t}.png')
            writer.append_data(image)
    for t in range(args.path_len):
        os.remove(f'gif_scratch/frame{t}.png')
os.system('rm -rf gif_scratch')
