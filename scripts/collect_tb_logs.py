"""
Collect the tensorboard logs into one directory.

Author: Ian Char
Date: February 2, 2023
"""
import argparse
import os


###########################################################################
# %% Parse arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True, type=str)
parser.add_argument('--save_dir', default='./logs', type=str)
args = parser.parse_args()

###########################################################################
# %% Make collections.
###########################################################################
os.makedirs(args.save_dir, exist_ok=True)
for root, dirs, files in os.walk(args.base_dir):
    to_path = None
    tb_in_root = False
    for fname in files:
        if 'events.out.tfevents' in fname:
            if to_path is None:
                to_path = os.path.join(args.save_dir, root)
                os.makedirs(to_path, exist_ok=True)
            from_path = os.path.join(root, fname)
            fpath = os.path.join(to_path, fname)
            os.system(f'cp {from_path} {fpath}')
print('Done!')
