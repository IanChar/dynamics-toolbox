import os
import pathlib

DYNAMICS_TOOLBOX_PATH = str(pathlib.Path(__file__).parent.parent.resolve())
os.environ['DYNAMICS_TOOLBOX_PATH'] = DYNAMICS_TOOLBOX_PATH
