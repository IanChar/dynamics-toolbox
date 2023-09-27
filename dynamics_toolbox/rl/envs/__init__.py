from gym.envs.registration import register

from dynamics_toolbox.rl.envs.cartpole import CartPole
from dynamics_toolbox.rl.envs.beta_tracking import BetaTracking


register(
    id='CtsCartPole-v0',
    entry_point=CartPole
)

register(
    id='BetaTracking-v0',
    entry_point=BetaTracking
)
