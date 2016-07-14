import numpy as np
import tensorflow as tf
from vizbot.core import Agent
from vizbot.utility import AttrDict


class DQN(Agent):

    def __init__(self, env, config=None):
        super.__init__(env)
        self._config = config or self._default_config()

    def begin(self):
        super().begin()
        self._input = collections.deque(
            maxlen=self._config.input_window)
        self._experience = collections.deque(
            maxlen=self._config.replay_capacity)

    def step(self, state):
        super.__call__()
        self._input.append(state)
        epsilon = self._decay(**self._config.epsilon)

    def feedback(self, reward):
        super().feedback(reward)

    @staticmethod
    def _default_config():
        input_window=4
        replay_capacity = int(1e6)
        batch_size = 32
        optimizer = tf.train.RMSPropOptimzier()
        epsilon = AttrDict(start=1, end=0.1, over=int(1e6))
        return AttrDict(**locals())
