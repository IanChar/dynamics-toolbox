# dynamics-toolbox
Toolbox for easily training and using dynamics models.

## Set Up

```
git clone https://github.com/IanChar/dynamics-toolbox.git
cd dynamics-toolbox
git fetch origin
git switch dev
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

### Anatomy of Config

### List of Models

## Data Generation

The data to be trained on is expected to be in an hdf5 file containing a dictionary with five different numpy arrays:
* observations with shape (num_samples, obs_dim)
* actions with shape (num_samples, act_dim)
* next_observations with shape (num_samples, obs_dim)
* rewards with shape (num_samples, 1)
* terminals with shape (num_sampes, 1)

Where necessary the code will parse these by trajectories (i.e. if the model considers sequences of data instead of individual next transition predictions)
