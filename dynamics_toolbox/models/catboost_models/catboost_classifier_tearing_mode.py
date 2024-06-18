"""
Catboost classifier model for tearing mode prediction.
The model is trained in FusionControl repo. 

Author: Rohit Sonker
"""
from catboost import CatBoostClassifier
import numpy as np

from typing import Sequence, Tuple, Dict, Any, Optional, Callable

from dynamics_toolbox.models.catboost_models.abstract_catboost_model import AbstractCatboostModel
import dynamics_toolbox.constants.losses as losses

class CatBoostTearingModeModel():
    """ Catboost classifier. """

    def __init__(
            self,
            model_path: str,
            # learning_rate: float = None,
            # depth: int = 5,
            # loss_function: str = losses.CB_LL,
            # # loss_type: str = losses.CE,
            # weight_decay: Optional[float] = 0.0,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension, equal to number of classes
            model_path: The directory to load the model from
            learning_rate: The learning rate for the network.
            depth: The number of .
            layer_size: The size of each hidden layer in the MLP.
            architecture: The architecture of the MLP described as a
                a string of underscore separated ints e.g. 256_100_64.
                If provided, this overrides num_layers and layer_sizes.
            hidden_activation: Activation to use.
            loss_type: The name of the loss function to use.
            weight_decay: Goes into l2_leaf_reg
            class_weights

        """
        self._model = CatBoostClassifier()
        self._model.load_model(model_path)
        self.input_states = [
        'betan_EFIT01',
        'dssdenest',
        'li_EFIT01',
        'q0_EFIT01',
        'q95_EFIT01',
        #  'n1rms',
        #  'n2rms',
        #  'n3rms',
        'vloop',
        'wmhd_EFIT01',
        #  'tm_label',
        'temp_component1',
        'temp_component2',
        'temp_component3',
        'temp_component4',
        'itemp_component1',
        'itemp_component2',
        'itemp_component3',
        'itemp_component4',
        'dens_component1',
        'dens_component2',
        'dens_component3',
        'dens_component4',
        'rotation_component1',
        'rotation_component2',
        'rotation_component3',
        'rotation_component4',
        'pres_EFIT01_component1',
        'pres_EFIT01_component2',
        'q_EFIT01_component1',
        'q_EFIT01_component2'
        ]

        self.input_actuators = [
        'pinj',
        'tinj',
        'ipsiptargt',
        'dstdenp',
        'bt_magnitude',
        'bt_is_positive',
        'D_tot',
        'aminor_EFIT01',
        'tritop_EFIT01',
        'tribot_EFIT01',
        'kappa_EFIT01',
        'rmaxis_EFIT01',
        'zmaxis_EFIT01',
        'ec_qrfe_param1',
        'ec_qrfe_param2',
        'ec_qrfe_param3'
        ]

    
    def get_prediction(self, state, action, info, **kwargs):
        # the action received will have full action if defined in reward function

        # get states as per states selected
        input_states_indices = [info['state_space'].index(s) for s in self.input_states]
        input_actuators_indices = [info['actuator_space'].index(a) for a in self.input_actuators]
        state = state[:, input_states_indices]
        if action.ndim==1:
            action = action.reshape(1,-1)
        action = action[:, input_actuators_indices]
        # next_action_delta = next_action_delta[:, input_actuators_indices]
        # full_action = action + next_action_delta

        model_input = np.concatenate((state, action), axis=1)

        pred = self._model.predict_proba(model_input)[:,1]

        return pred

    