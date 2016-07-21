from .simple_doom import SimpleDoom
from .simple_atari import SimpleAtari
from gym.envs.registration import register


register(
    id='SimpleDoom-v0',
    entry_point='vizbot.env:SimpleDoom',
    timestep_limit=10000,
    kwargs=dict(env='DoomDeathmatch-v0'),
)

register(
    id='SimpleAtari-v0',
    entry_point='vizbot.env:SimpleAtari',
    timestep_limit=10000,
    kwargs=dict(env='Breakout-v0'),
)
