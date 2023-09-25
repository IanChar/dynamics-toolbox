import gym

from dynamics_toolbox.rl.modules.policies.mpc_policy import MPCPolicy
from dynamics_toolbox.env_wrappers.model_env import ModelEnv
from dynamics_toolbox.utils.storage.model_storage import load_model_from_log_dir
from dynamics_toolbox.rl.util.gym_util import evaluate_policy_in_gym


model = load_model_from_log_dir(
    path='/zfsauton/project/public/ichar/AutoCal/models/d4rl/halfcheetah_medium_replay-v0/0/0',
)
model_env = ModelEnv(model)
env = gym.make('HalfCheetah-v2')
policy = MPCPolicy(
    model_env=model_env, 
)
evaluate_policy_in_gym(env, policy, num_eps=1)