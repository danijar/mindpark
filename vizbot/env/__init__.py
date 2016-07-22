from .simple_doom import SimpleDoom
from .simple_atari import SimpleAtari
from gym.envs.registration import register


register(
    id='SimpleDeathmatch-v0',
    entry_point='vizbot.env:SimpleDoom',
    timestep_limit=10000,
    kwargs=dict(env='DoomDeathmatch-v0'),
)

register(
    id='SimplePong-v0',
    entry_point='vizbot.env:SimpleAtari',
    timestep_limit=10000,
    kwargs=dict(env='Pong-v0'),
)

register(
    id='SimpleBreakout-v0',
    entry_point='vizbot.env:SimpleAtari',
    timestep_limit=10000,
    kwargs=dict(env='Breakout-v0'),
)
