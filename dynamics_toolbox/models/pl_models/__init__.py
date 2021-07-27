"""
Pytorch Lightning models.

Author: Ian Char
"""
from dynamics_toolbox.models.pl_models.fc_model import FcModel
from dynamics_toolbox.models.pl_models.pnn import PNN

PL_MODELS = {
    'FC': FcModel,
    'PNN': PNN,
}