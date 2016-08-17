from gym.envs.registration import register
from .simple_doom import SimpleDoom


register(
    id='SimpleDeathmatch-v0',
    entry_point='vizbot.env:SimpleDoom',
    timestep_limit=10000,
    kwargs=dict(env='DoomDeathmatch-v0'),
)

register(
    id='SimpleGather-v0',
    entry_point='vizbot.env:SimpleDoom',
    timestep_limit=10000,
    kwargs=dict(env='DoomHealthGathering-v0'),
)
