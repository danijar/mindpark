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
        self.__timestep = 0

    def perform(self, state):
        super().perform(state)
        self.__frames.append(state)
        self.__timestep += 1
        if self.__timestep < self.__width:
            return self._noop()
        super().step()
        frames = np.moveaxis(np.array(self.__frames), 0, -1)
        return self._agent.perform(frames)

    def feedback(self, action, reward):
        super().feedback(action, reward)
        self._agent.feedback(action, reward)
