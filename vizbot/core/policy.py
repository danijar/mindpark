from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class State(Enum):
    """
    State of the policy. Only used for validation.
    """
    initial = 1
    begin = 2
    end = 3
    observed = 4
    received = 5


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
        self._state = State.initial

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
        self._assert_state(State.initial, State.end)
        self._state = State.begin
        self.training = training

    def end_episode(self):
        """
        Optional hook at the end of an episode.
        """
        self._assert_state(State.begin, State.received)
        self._state = State.end
        self.training = None

    @abstractmethod
    def observe(self, observation):
        """
        Process an observation. This includes to experience the last transition
        and form an action.
        """
        self._assert_state(State.begin, State.received)
        self._state = State.observed
        assert self.observations.contains(observation)
        self.step = 0 if self.step is None else self.step + 1

    @abstractmethod
    def receive(self, reward, final):
        """
        Receive a reward from the environment that will later by used to
        experience the current transition. Preprocesses should forward the
        reward to the above policy where appropriate.
        """
        self._assert_state(State.observed)
        self._state = State.received
        assert reward is not None

    def _assert_state(self, *states):
        if self._state in states:
            return
        message = "{} should be in {} but is in '{}'"
        states = ' or '.join("'{}'".format(x.name) for x in states)
        message = message.format(type(self).__name__, states, self._state.name)
        raise RuntimeError(message)
