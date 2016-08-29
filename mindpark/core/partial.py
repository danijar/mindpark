from abc import abstractmethod
from mindpark.core.policy import Policy
from mindpark.core.task import Task
from mindpark.utility import Proxy


class Partial(Policy):

    """
    A way to interact with the environment that forwards some of the decisions
    to an above policy.

    Note: After forwarding an observation, you must forward a reward, before
    continuing with the next observation.
    """

    def __init__(self, task):
        super().__init__(task)
        self.above = None

    def set_above(self, above):
        assert above is not None
        if not self.above_task:
            raise ValueError('cannot add to final policy')
        self._validate_above(above)
        self.above = above

    @property
    def above_task(self):
        """
        The task for the above policy. Usually, you do not want to override
        this directly. Return None to indicate a full policy.
        """
        assert (self.above_observs is None) == (self.above_actions is None)
        if self.above_observs is None:
            return None
        task = Proxy(self.task)
        task.observs = self.above_observs
        task.actions = self.above_actions
        return task

    @property
    @abstractmethod
    def above_observs(self):
        """
        The space of observations passed to the above policy.
        """
        pass

    @property
    @abstractmethod
    def above_actions(self):
        """
        The space of actions expected from the above policy.
        """
        pass

    def begin_episode(self, episode, training):
        if self.above_task and self.above is None:
            raise RuntimeError('must set above policy before simulation')
        super().begin_episode(episode, training)

    def _validate_above(self, above):
        if self.above_task.step is not above.task.step:
            raise ValueError('above policy uses different step counter')
        if self.above_task.observs != above.task.observs:
            raise ValueError('above policy expects different observations')
        if self.above_task.actions != above.task.actions:
            raise ValueError('above policy expects different actions')
