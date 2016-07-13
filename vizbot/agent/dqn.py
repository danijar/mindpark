import numpy as np
import tensorflow as tf
from vizbot.core import Agent


class DQN(Agent):

    def __init__(self, env):
        self._env = env
        self._experience = []

    def __call__(self, reward, state):
        pass
