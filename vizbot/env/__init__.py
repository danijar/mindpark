from .simple_doom import SimpleDoom
from gym.envs.registration import register


register(
    id='SimpleDoom-v0',
    entry_point='vizbot.env:SimpleDoom',
    timestep_limit=10000,
    reward_threshold=1000.0,
)
