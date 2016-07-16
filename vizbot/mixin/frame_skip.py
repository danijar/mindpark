import numpy as np
from gym.spaces import Box
from vizbot.core import Mixin


class FrameSkip(Mixin):

    def __init__(self, agent_cls, states, actions, width=1):
        self._states = states
        self._actions = actions
        self._width = width
        super().__init__(agent_cls)
        self._action = self._agent._noop()
        self._frames = None
        self._index = None

    @property
    def states(self):
        states = Box(0, 255, self._states.shape + (self._width,))
        return states

    @property
    def actions(self):
        return self._actions

    def begin(self):
        self._agent.begin()
        self._frames = np.empty((self._width,) + self._states.shape)
        self._index = 0

    def step(self, state):
        self._frames[self._index % self._width] = state
        self._index += 1
        if self._index and self._index % self._width:
            frames = np.moveaxis(self._frames, 0, -1)
            self._action = self._agent.step(frames)
        return self._action
