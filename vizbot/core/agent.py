import numpy as np
import tensorflow as tf
import gym


class Agent:

    def __init__(self, env):
        self._env = env
        self._actions = env.action_space.shape
        self._states = env.observation_space.shape

    def begin(self):
        self._episode = -1

    def step(self, state):
        self._episode += 1

    def feedback(self, previous, action, reward, successor):
        pass

    def end(self):
        pass

    def _noop(self):
        return np.zeros(self._env.action_space.shape)

    def _decay(self, start, end, over):
        progress = min(self._episode, over) / over
        return (1 - progress) * start + progress * end
