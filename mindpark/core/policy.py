from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from mindpark.utility import Proxy


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
    A way to interact with an environment.
    """

    def __init__(self, task):
        """
        The task specifies the interface of the environment, a counter of the
        underlying environment step, a directory to store logs and checkpoints,
        and more.
        """
        self.task = Proxy(task)
        self.random = np.random.RandomState()
        self.training = None
        self._state = State.initial

    def begin_episode(self, episode, training):
        """
        Optional hook at the beginning of an episode. The `training` parameter
        specifies whether the policy should learn during the episode or is just
        evaluated.
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
    def observe(self, observ):
        """
        Process an observation and return an action.
        """
        self._assert_state(State.begin, State.received)
        self._state = State.observed
        if not self.task.observs.contains(observ):
            message = '{} received an invalid observation'
            raise ValueError(message.format(self))

    @abstractmethod
    def receive(self, reward, final):
        """
        Receive a reward from the environment.
        """
        self._assert_state(State.observed)
        self._state = State.received
        assert reward is not None

    def __repr__(self):
        return '<{}>'.format(type(self).__name__)

    def _assert_state(self, *states):
        if self._state in states:
            return
        message = "{} should be in {} but is in '{}'"
        states = ' or '.join("'{}'".format(x.name) for x in states)
        message = message.format(type(self).__name__, states, self._state.name)
        raise RuntimeError(message)
