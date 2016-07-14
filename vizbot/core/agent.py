import numpy as np
import tensorflow as tf
import gym


class Agent:

    def __init__(self, env):
        self._env = env

    def begin(self):
        self._episode = -1

    def step(self, state):
        self._episode += 1

    def feedback(self, reward):
        pass

    def end(self):
        pass

    def _decay(self, start, end, over):
        progress = min(self._episode, over) / over
        return (1 - progress) * start + progress * end
