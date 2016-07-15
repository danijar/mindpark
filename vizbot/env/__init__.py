from .doom_simple_deathmatch import DoomSimpleDeathmatch
from gym.envs.registration import register


register(
    id='DoomSimpleDeathmatch-v0',
    entry_point='vizbot.env:DoomSimpleDeathmatch',
    timestep_limit=10000,
    reward_threshold=1000.0,
)
