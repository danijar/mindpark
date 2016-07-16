import collections
import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class FrameWindow(Preprocess):

    def __init__(self, env, width):
        super().__init__(env)
        self.__width = width
        self.__frames = None

    @property
    def states(self):
        shape = self._env.states.shape + (self.__amount,)
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def start(self):
        super().start()
        self.__frames = collections.deque(maxlen=self.__width)
        self.__index = 0

    def perform(self, state):
        super().perform(state)
        self.__frames.append(state)
        self.__index += 1
        frames = np.moveaxis(np.array(self.__frames), 0, -1)
        return self._agent.perform(frames)
