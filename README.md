# Dyanmics Toolbox
Toolbox for easily training dynamics models, generating predictions, and doing
model-based reinforcement learning.

This library was motivated by the fact that many model-based reinforcement
learning works do not do much exploration when it comes to the dynamics models
being used. Most of the times these models are simply ensembles of "PNNs". As such,
this library provides the flexibility to do model-based reinforcement learning using
a wide variety of different neural network models.

# Authors
* [Ian Char](ianchar.com)
* [Youngseog Chung](https://youngseogchung.github.io/)
* [Rohan Shah](https://www.linkedin.com/in/rohan-shah13/)
* [Viraj Mehta](https://virajm.com/)

## Set Up

```
git clone https://github.com/IanChar/dynamics-toolbox.git
cd dynamics-toolbox
```

Make a virtual environment with python=3.9 and then run

```
pip install -r requirements.txt
pip install -e .
```

## Model Training

Training a model can be done through [train.py](./train.py). The code relies heavily on [hydra](https://hydra.cc/), and all options should be specified in .yaml config files. You can use [example_configs](./example_configs) for reference.

As an initial test you can run
```
python train.py -cn example_rnn cuda_device=0
```
where the cuda_device is optional. This will train a recurrent neural network on pendulum data.

### Loading and Using Models

To load a model there are several different utility functions in [model_storage](./dynamics_toolbox/utils/storage/model_storage.py]. For example if you know the path to the directory a model was saved to, you can load in the model with
```
model = load_model_from_log_dir(path=<path_to_model_dir>)
```
Once the model is imported you can predict next states by
```
model.reset()
pred, pred_info_dict = model.predict(np.concatenate([observations, actions], dim=-1]))
```
making sure to call model.reset() whenever there is a new trajectory being predicted for.

### Model Implementation

Almost all models inherit from (AbstractPlModel)[dynamics_toolbox/models/pl_models/abstract_pl_model.py).
There are a few important methods that new models should implement in this class.

* get_net_out: This is a method that takes in a training batch and outputs a dictionary of information that will be needed by the loss step. One of these outputs ought to be "prediction" in order to calculate metrics when logging.

* loss: This takes the dictionary outputted by get_net_out and calculates the loss. It returns both the loss and a dictionary of other statistics we may want to log.

* single_sample_output_from_torch: Assuming that the model learned has a distribution over different functions (e.g. Ensemble, SimplexMLP, NeuralProcess, etc), this function makes predictions where all predictions come from a single function sampled from the distribution. If the modle forms a predictive distribution but cannot draw coherent, smooth function samples (e.g. a Gaussian is predicted), then this function whill behave the same as multi_sample_output_from_torch. The predict function will use either this method or the multi version depending on whether each_input_is_different_sample is set to True or not.

* multi_sample_output_from_torch: Samples from the distribution assuming that input in net_in should come from a different function sample.

* metrics: A dictionary of functions that computes the metrics that should be logged.

* learning_rate: Returns the learning rate to set in the optimizer.

* weight_decay: Returns the weight decay to set in the optimizer.

### Ensembling Models

To ensemble models, one can use the [Ensemble](dynamics/models/ensemble.py) class.
If loading these from previously saved models, you can use the helper function
load_ensemble_from_parent_dir found
[here](dynamics_toolbox/utils/storage/model_storage.py).

### Making Configurations for Training

This library relies heavily on [Hydra](https://hydra.cc) for configuration. Each
configuration for model training must have a few different. The main ones are:

* name: The name of the run.
* data_source: Path for the data.
* model: A sub-config that specifies the model.
* trainer: A sub-config that specifies training parameters.
* data_module: A sub-config that specifies how to set up the lightning data loader.
* early_stopping: A sub-config that specifies how to stop training early.

For examples look at the example_configs directory.

### List of Models

The full list of models can be found in dynamocs_toolbox/models/pl_models.

#### Standard Models

* [MLP](dynamics_toolbox/models/pl_models/mlp.py) A standard fully connected neural network.
* [ResidualMLPBlocks](dynamics_toolbox/models/pl_models/residual_mlp_blocks.py) A neural network architecture with skip connections.
* [PNN](dynamics_toolbox/models/pl_models/pnn.py) A model that predicts a Gaussian distribution. Note that this architecture can be built upon the previously listed MLPs.
* [QuantileModel](dynamics_toolbox/models/pl_models/quantile_model.py) A model outputs the quantiles of a predictive distribution.
* [DropoutMLP](dynamics_toolbox/models/pl_models/droppout_mlp.py) A model that uses dropout masks for uncertainty.
* [SimplexMLP](dynamics_toolbox/models/pl_models/simplex_mlp.py) A model which learns a subspace of good neural network parameters.
* [SimultaneousEnsemble](dynamics_toolbox/models/pl_models/simultaneous_ensemble.py) A model which trains a neural network ensemble at the same time. This implementation allows for a diversity term to be included across all ensemble members so that the network weights are diverse.

#### Sequential Models

These are models that incorporate some history of observations made previously.

* [LinearAutoregress](dynamics_toolbox/models/pl_models/sequential_models/linear_autoregress.py) A linear model that predicts next labels through a linear combination of past labels.
* [RNN](dynamics_toolbox/models/pl_models/sequential_models/rnn.py) A standard recurrent network that uses either LSTMs or GRUs. By default this does not autoregress when training.
* [RPNN](dynamics_toolbox/models/pl_models/sequential_models/rpnn.py) A recurrent neural network that predicts a Gaussian distribution for the label.
* [GPT](dynamics_toolbox/models/pl_models/sequential_models/transformers/gpt.py) A GPT-style transformer neural network.

#### Sequential Dynamics Models (BETA)

These are models that train autoregressively and therefore need to separate between
the states and actions in the input to the model. They can be found
[here](dynamics_toolbox/models/pl_models/sequential_models/sequential_dynamics_models)

CAUTION: There may be errors using these models!

#### Conditional Models (BETA)

These are models that adjust the outputted distribution based on previous observations
made (e.g. Neural Processes). These models can be found
[here](dynamics_toolbox/models/pl_models/conditional_models).

CAUTION: There may be errors using these models!

## Data Management and Formatting

### Data Modules

There are a few data modules that one can use depending on the type of data and the
type of model being trained.

* [RegressionDataModule](dynamics_toolbox/data/pl_data_modules/regression_data_module.py) Use this data module when you want to do standard regression. This data module expects the data to be presented as x, y information.

* [ForwardDynamicsDataModule](dynamics_toolbox/data/pl_data_modules/forward_dynamics_data_module.py) Use this data module for standard dynamics modeling where history is not needed. This assumes the data is structured as observations, actions, next_observations, and possibly rewards.

* [SequantialDataModule](dynamics_toolbox/data/pl_data_modules/forward_dynamics_data_module.py) Data module for dynamics where the model needs this history (e.g. RNN). The data file must have the data listed in order for this to work correctly.

### Data Generation

The data to be trained on is expected to be in an hdf5 file containing a dictionary with five different numpy arrays:
* observations with shape (num_samples, obs_dim)
* actions with shape (num_samples, act_dim)
* next_observations with shape (num_samples, obs_dim)
* rewards with shape (num_samples, 1)
* terminals with shape (num_sampes, 1)

Where necessary the code will parse these by trajectories (i.e. if the model considers sequences of data instead of individual next transition predictions)

## Reinforcement Learning

We provide a number of ways of learning a policy via reinforcement learning with
and without learned surrogate models. As of now we only provide Soft Actor Critic (SAC)
for policy learning algorithms. To train the policy run

```
python train_rl.py
```

There are a few degrees of freedom here, and example configurations can be found in
example_configs/rl.

* The policy can either get experience from the real environment or the model.
* The policy can be either an MLP or a recurrent neural network.
* The policy can either collect additional information or be trained in the offline setting.

Depending on the options selected, we can replicate different algorithms:

* [MBPO](https://arxiv.org/abs/1906.08253.pdf) = Model-based Online RL with MLP policy.
* [MOPO](https://arxiv.org/abs/2005.13239) = Model-based Offline RL with MLP policy.
* [MAPLE](https://proceedings.neurips.cc/paper/2021/hash/470e7a4f017a5476afb7eeb3f8b96f9b-Abstract.html) = Model-based Offline RL with recurrent policy.

Comparing against the results reported by these papers, our implementation gets
comporable (though very slightly worse) results for MOPO. We also get somewhat comporable
results for all results in MAPLE except for the walker environment, which we our
implementation is noticeably worse for.

## Utility Scripts

There are several utility scripts found in the scripts directory that may be of
interest. Perhaps most interesting are the model [diagnostics](scripts/model_diagnostics).
There are several scripts here that evaluate and visualize the model's uncertainty
over time.

## Citations

If you find this library useful, please consider citing one of the works that lead
to its development.

* This library was used to model plasma dynamics and trained policies that were
ultimately deployed on the DIII-D device.
```
@inproceedings{char2023offline,
  title={Offline model-based reinforcement learning for tokamak control},
  author={Char, Ian and Abbate, Joseph and Bard{\'o}czi, L{\'a}szl{\'o} and Boyer, Mark and Chung, Youngseog and Conlin, Rory and Erickson, Keith and Mehta, Viraj and Richner, Nathan and Kolemen, Egemen and others},
  booktitle={Learning for Dynamics and Control Conference},
  pages={1357--1372},
  year={2023},
  organization={PMLR}
}
```

* This library was used for a workshop which questioned the use of the sampling scheme for PNNs. We instead propose a way to sample smooth functions from PNNs.
```
@inproceedings{char2023correlated,
  title={Correlated Trajectory Uncertainty for Adaptive Sequential Decision Making},
  author={Char, Ian and Chung, Youngseog and Shah, Rohan and Neiswanger, Willie and Schneider, Jeff},
  booktitle={NeurIPS 2023 Workshop on Adaptive Experimental Design and Active Learning in the Real World},
  year={2023}
}
```
