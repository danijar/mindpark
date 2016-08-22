import inspect
from vizbot.core.partial import Partial


class Sequential(Partial):

    def __init__(self, task):
        super().__init__(task)
        self.steps = []

    def add(self, policy, *args, **kwargs):
        """
        Add a policy to the sequential behavior. If the policy provides no
        interface, no further policies can be added.
        """
        if inspect.isclass(policy):
            policy = policy(self.above_task, *args, **kwargs)
        elif args or kwargs:
            raise ValueError('unexpected args for an instantiated policy')
        if self.steps:
            self.steps[-1].set_above(policy)
        self.steps.append(policy)
        if self.above:
            policy.set_above(self.above)

    @property
    def above_task(self):
        if not self.steps:
            return self.task
        if hasattr(self.steps[-1], 'above_task'):
            return self.steps[-1].above_task
        return None

    @property
    def above_observs(self):
        pass

    @property
    def above_actions(self):
        pass

    def set_above(self, above):
        super().set_above(above)
        if self.steps:
            self.steps[-1].set_above(above)

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        for policy in self.steps:
            policy.begin_episode(episode, training)

    def end_episode(self):
        for policy in reversed(self.steps):
            policy.end_episode()
        super().end_episode()

    def observe(self, observ):
        super().observe(observ)
        return self._first.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        self._first.receive(reward, final)

    def __repr__(self):
        steps = ', '.join([type(x).__name__ for x in self.steps])
        return "<Sequential steps=[{}]>".format(steps)

    @property
    def _first(self):
        if self.steps:
            return self.steps[0]
        return self.above
