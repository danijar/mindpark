import numpy as np
from gym.spaces import Box
from vizbot.core import Mixin


class Image(Mixin):

    def __init__(self, agent_cls, states, actions, mode='color', factor=1):
        self._states = states
        self._actions = actions
        self._mode = mode
        self._factor = factor
        super().__init__(agent_cls)

    @property
    def states(self):
        shape = list(self._states.shape)
        if self._mode == 'grayscale':
            shape = shape[: -1]
        shape[0] //= self._factor
        shape[1] //= self._factor
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._actions

    def step(self, state):
        state = self._color(state, self._mode)
        state = self._subsample(state, self._factor)
        state = state.astype(np.uint8)
        return self._agent.step(state)

    @staticmethod
    def _color(state, mode):
        if mode == 'color':
            return state
        elif mode == 'grayscale':
            return state.mean(-1)
        raise NotImplementedError

    @staticmethod
    def _subsample(state, factor):
        if not factor or factor == 1:
            return state
        if not isinstance(factor, int):
            raise NotImplementedError
        width, height = state.shape[:2]
        shape = width // factor, factor, height // factor, factor, -1
        state = state.reshape(shape).mean(3).mean(1)
        state = np.squeeze(state)
        return state
