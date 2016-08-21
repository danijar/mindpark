from abc import abstractmethod
from vizbot.core.policy import Policy
from vizbot.core.task import Task


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
        self._validate_task(above.task)
        self.above = above

    @property
    def above_task(self):
        """
        The task for the above policy. Usually, you do not want to override
        this directly. Return None to indicate a full policy.
        """
        task = Task(
            self.above_observs, self.above_actions, self.task.directory,
            self.task.steps, self.task.step, self.task.episode)
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

    def _validate_task(self, task):
        if self.above_task.step is not task.step:
            raise ValueError('above policy uses different step counter')
        if self.above_task.observs != task.observs:
            raise ValueError('above policy expects different observations')
        if self.above_task.actions != task.actions:
            raise ValueError('above policy expects different actions')
