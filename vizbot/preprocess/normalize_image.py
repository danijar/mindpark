import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class NormalizeImage(Preprocess):

    def __init__(self, env):
        super().__init__(env)
        self._low = env.states.low
        self._high = env.states.high

    @property
    def states(self):
        return Box(0.0, 1.0, self._env.states.shape)

    @property
    def actions(self):
        return self._env.actions

    def reset(self):
        state = self._env.reset()
        state = self._apply(state)
        return state

    def step(self, action):
        state, reward = self._env.step(action)
        state = self._apply(state)
        return state, reward

    def _apply(self, state):
        state = state.astype(float)
        state = (state - self._low) / (self._high - self._low)
        return state
