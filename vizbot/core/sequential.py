import inspect
from vizbot.core.policy import Policy


class Sequential(Policy):

    def __init__(self, interface):
        super().__init__(interface)
        self.steps = []

    def add(self, policy, *args, **kwargs):
        """
        Add a policy to the sequential behavior. If the policy provides no
        interface, no further policies can be added.
        """
        if self.interface is None:
            raise RuntimeError('cannot add to final policy')
        if inspect.isclass(policy):
            policy = policy(self.interface, *args, **kwargs)
        elif args or kwargs:
            raise ValueError('unexpected args for an instantiated policy')
        if self.steps:
            self.steps[-1].set_above(policy)
        if self.above:
            policy.set_above(self.above)
        self.steps.append(policy)

    @property
    def interface(self):
        if not self.steps:
            return self.observations, self.actions
        return self.steps[-1].interface

    def set_above(self, above):
        super().set_above(above)
        if self.steps:
            self.steps[-1].set_above(self.above)

    def begin_episode(self, training):
        super().begin_episode(training)
        for policy in self.steps:
            policy.begin_episode(training)

    def end_episode(self):
        super().end_episode()
        for policy in self.steps:
            policy.end_episode()

    def observe(self, observation):
        super().observe(observation)
        return self._first.observe(observation)

    def receive(self, reward, final):
        super().receive(reward, final)
        self._first.receive(reward, final)

    @property
    def _first(self):
        if self.steps:
            return self.steps[0]
        elif not self.above:
            raise RuntimeError('need to know above policy before simulation')
        return self.above
