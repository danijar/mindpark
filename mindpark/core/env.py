from abc import ABC, abstractmethod


class Env(ABC):

    @property
    @abstractmethod
    def observs(self):
        """
        The observation space.
        """
        pass

    @property
    @abstractmethod
    def actions(self):
        """
        The observation space.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Initialize or reinitialize the environment. Return an initial
        observation.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Apply the action and simulate one time step in the environment. Return
        the reward and next observation. May return None as observation to stop
        the episode.
        """
        pass
