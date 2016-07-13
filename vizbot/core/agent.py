from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import gym


class Agent(ABC):

    @abstractmethod
    def __init__(self, env):
        pass

    @abstractmethod
    def __call__(self, reward, state):
        pass
