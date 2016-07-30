import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class FrameSkip(Preprocess):

    def __init__(self, env, amount):
        super().__init__(env)
        low, high = self._env.states.low, self._env.states.high
        self._low, self._high = low.flatten()[0], high.flatten()[0]
        assert (low == self._low).all()
        assert (high == self._high).all()
        self._amount = amount
        self._frames = np.zeros((self._amount,) + self._env.states.shape)

    @property
    def states(self):
        shape = self._env.states.shape + (self._amount,)
        return Box(self._low, self._high, shape)

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
