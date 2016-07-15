import numpy as np
from gym.spaces import HighLow
from gym.envs.doom.doom_deathmatch import DoomDeathmatchEnv
from doom_py import ScreenResolution


class DoomSimpleDeathmatch(DoomDeathmatchEnv):

    ALLOWED_ACTIONS = [
        0,                       # Fire
        4,                       # Turn 180
        8,                       # Speed
        9,                       # Move sideways in a circle
        10, 11, 12, 13,          # Move
        14, 15,                  # Turn
        21, 22, 23, 24, 26, 27,  # Select weapon
        38, 39,                  # Mouse look
    ]

    def __init__(self):
        super().__init__()
        self.configure(screen_resolution=ScreenResolution.RES_160X120)

    def _load_level(self):
        result = super()._load_level()
        self.full_action_space = self.action_space
        self._allowed_actions = sorted(self.ALLOWED_ACTIONS)
        matrix = self.action_space.matrix[self._allowed_actions]
        self.action_space = HighLow(matrix)
        return result

    def _step(self, action):
        assert len(action) == len(self._allowed_actions)
        full_action = np.zeros(self.full_action_space.num_rows)
        for index, value in zip(self._allowed_actions, action):
            full_action[index] = value
        state, reward, done, info = super()._step(full_action)
        state = self._downsample(state)
        return state, reward, done, info

    def _render(self, *args, **kwargs):
        image = super()._render(*args, **kwargs)
        if isinstance(image, np.ndarray):
            image = self._downsample(image)
        return image

    @staticmethod
    def _downsample(state, factor=2):
        width, height, _ = state.shape
        shape = width // factor, factor, height // factor, factor, -1
        state = state.reshape(shape).mean(3).mean(1)
        state = state.astype(np.uint8)
        return state
