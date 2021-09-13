"""
Configurations for training.
"""
import dynamics_toolbox.constants.activations as activations

CONFIGS = {}
CONFIGS['fc'] = {
   'model_type': 'FC',
   'framework': 'pytorch-lightning',
   'training_type': 'standard',
   'hidden_sizes': '200,200,200,200',
   'learning_rate': 1e-3,
   'batch_size': 256,
   'val_proportion': 0.1,
   'patience': 10,
}
CONFIGS['fc_ensemble'] = {
   'model_type': 'FC',
   'framework': 'pytorch-lightning',
   'num_ensemble_members': 5,
   'training_type': 'standard',
   'hidden_sizes': '200,200,200,200',
   'learning_rate': 1e-3,
   'batch_size': 256,
   'val_proportion': 0.1,
   'patience': 10,
}
# Configuration for how models are trained in MOPO: https://arxiv.org/abs/2005.13239
CONFIGS['pets'] = {
   'model_type': 'PNN',
   'framework': 'pytorch-lightning',
   'num_ensemble_members': 7,
   'training_type': 'standard',
   'encoder_hidden_sizes': '200,200,200',
   'encoder_output_dim': 200,
   'mean_hidden_sizes': '',
   'logvar_hidden_sizes': '',
   'logvar_lower_bound': -10,
   'logvar_upper_bound': 0.5,
   'logvar_bound_loss_coef': 1e-2,
   'learning_rate': 1e-3,
   'batch_size': 256,
   'val_proportion': 0.1,
   'hidden_activation': activations.SWISH,
   'patience': 10,
   'min_delta': 1e-2,
}
