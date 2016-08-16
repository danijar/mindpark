from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):

    """
    A behavior to interact with an environment. Call super when overriding
    methods.

    The behavior can be partial and delegate decisions to the next policy. In
    this case, override `interface` and use `above` to access the next policy.
    Consider forwarding observations from `perform()` to `above.observe()` and
    rewards from `receive()` to `above.receive()`.
    """

    def __init__(self, interface):
        """
        Construct a policy that interacts with an environment using the
        specified interface.
        """
        self.observations, self.actions = interface
        self.random = np.random.RandomState()
        self.training = None
        self.above = None
        self.step = None
        self._observe_or_receive = True

    @property
    def interface(self):
        """
        The interface for higher policies to interact with this one. The
        interface is a tuple of the observation and the action space. For final
        steps, the interface is None.
        """
        return None

    def set_above(self, above):
        """
        The above policy will be set through this setter. Its observation and
        action spaces will match the this policy's interface.
        """
        if self.interface is None:
            raise RuntimeError('cannot set above of final policy')
        assert self.interface[0] == above.observations
        assert self.interface[1] == above.actions
        self.above = above

    def begin_episode(self, training):
        """
        Optional hook at the beginning of an episode. The `training` parameter
        specifies whether the algorithm should learn during the episode or the
        algortihm is just evaluated.
        """
        self.training = training

    def end_episode(self):
        """
        Optional hook at the end of an episode.
        """
        self.training = None

    @abstractmethod
    def observe(self, observation):
        """
        Process an observation. This includes to experience the last transition
        and form an action.
        """
        assert self.observations.contains(observation)
        self.step = 0 if self.step is None else self.step + 1
        assert self._observe_or_receive, 'must receive reward first'
        self._observe_or_receive = False

    @abstractmethod
    def receive(self, reward, final):
        """
        Receive a reward from the environment that will later by used to
        experience the current transition. Preprocesses should forward the
        reward to the above policy where appropriate.
        """
        assert reward is not None
        assert not self._observe_or_receive, 'must observe environment first'
        self._observe_or_receive = True
