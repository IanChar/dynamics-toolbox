import gym
from gym.envs.registration import register
import pybullet_envs

from dynamics_toolbox.rl.envs.pybullet.pybullet_wrappers import POMDPWrapper

# Register all the partially observable pybullet environments.
# Taken from https://github.com/twni2016/pomdp-baselines
"""
The observation space can be divided into several parts:
np.concatenate(
[
    z - self.initial_z, # pos
    np.sin(angle_to_target), # pos
    np.cos(angle_to_target), # pos
    0.3 * vx, # vel
    0.3 * vy, # vel
    0.3 * vz, # vel
    r, # pos
    p # pos
], # above are 8 dims
[j], # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
[self.feet_contact], # depends on foot_list, belongs to pos
])
"""
register(
    "HopperBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HopperBulletEnv-v0"),
        partially_obs_dims=list(range(15)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "HopperBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HopperBulletEnv-v0"),
        partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14],  # one foot
    ),  # pos
    max_episode_steps=1000,
)

register(
    "HopperBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HopperBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13],
    ),  # vel
    max_episode_steps=1000,
)

register(
    "WalkerBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("Walker2DBulletEnv-v0"),
        partially_obs_dims=list(range(22)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "WalkerBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("Walker2DBulletEnv-v0"),
        partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21],  # 2 feet
    ),  # pos
    max_episode_steps=1000,
)

register(
    "WalkerBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("Walker2DBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
    ),  # vel
    max_episode_steps=1000,
)

register(
    "AntBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("AntBulletEnv-v0"),
        partially_obs_dims=list(range(28)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "AntBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("AntBulletEnv-v0"),
        partially_obs_dims=[
            0,
            1,
            2,
            6,
            7,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            25,
            26,
            27,
        ],  # 4 feet
    ),  # pos
    max_episode_steps=1000,
)

register(
    "AntBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("AntBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19, 21, 23],
    ),  # vel
    max_episode_steps=1000,
)

register(
    "HalfCheetahBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HalfCheetahBulletEnv-v0"),
        partially_obs_dims=list(range(26)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "HalfCheetahBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HalfCheetahBulletEnv-v0"),
        partially_obs_dims=[
            0,
            1,
            2,
            6,
            7,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
        ],  # 6 feet
    ),  # pos
    max_episode_steps=1000,
)

register(
    "HalfCheetahBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HalfCheetahBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
    ),  # vel
    max_episode_steps=1000,
)
