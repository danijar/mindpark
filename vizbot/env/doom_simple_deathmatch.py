import numpy as np
from gym.spaces import HighLow
from gym.envs.doom.doom_deathmatch import DoomDeathmatchEnv
from doom_py import ScreenResolution


class DoomSimpleDeathmatch(DoomDeathmatchEnv):

    ALLOWED_ACTIONS = [
        0,                           # Fire
        4,                           # Turn 180
        8,                           # Speed
        9,                           # Move sideways in a circle
        10, 11, 12, 13,              # Move
        14, 15,                      # Turn
        21, 22, 23, 24, 25, 26, 27,  # Select weapon
        38, 39,                      # Mouse look
    ]

    def __init__(self):
        super().__init__()
        self.configure(screen_resolution=ScreenResolution.RES_160X120)
        self.full_action_space = self.action_space
        self._reduce_action_space()

    def _load_level(self, *args, **kwargs):
        result = super()._load_level(*args, **kwargs)
        self._reduce_action_space()
        return result

    def _step(self, action):
        assert len(action) == len(self._allowed_actions)
        full_action = np.zeros(self.full_action_space.num_rows)
        for index, value in zip(self._allowed_actions, action):
            full_action[index] = value
        # Work around drop weapon bug in parent class.
        full_action[33] = 0
        return super()._step(full_action)

    def _reduce_action_space(self):
        self._allowed_actions = sorted(self.ALLOWED_ACTIONS)
        matrix = self.full_action_space.matrix[self._allowed_actions]
        self.action_space = HighLow(matrix)
