import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class FrameSkip(Preprocess):

    def __init__(self, env, amount):
        super().__init__(env)
        self._amount = amount
        self._frames = np.zeros((self._amount,) + self._env.states.shape)

    @property
    def states(self):
        shape = self._env.states.shape + (self._amount,)
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def reset(self):
        self._frames[0] = self._env.reset()
        for index in range(1, self._amount):
            state, _ = self._env.step(self._env.actions.sample())
            self._frames[index] = state
        return np.moveaxis(self._frames, 0, -1)

    def step(self, action):
        rewards = 0
        for index in range(self._amount):
            state, reward = self._env.step(action)
            self._frames[index] = state
            rewards += reward
        return np.moveaxis(self._frames, 0, -1), rewards
