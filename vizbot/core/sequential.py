import inspect
from vizbot.core.policy import Policy


class Step(Policy):

    """
    A partial policy that may delegate decisions to the next policy using the
    `above` property. Consider overriding `step()` and `experience()` to
    forward calls to `above` when appropriate.
    """

    def __init__(self, interface):
        super().__init__(interface)
        self._above = None

    @property
    def interface(self):
        """
        The interface for higher policies to interact with this one. The
        interface is a tuple of the observation and the action space.
        """
        return self.observations, self.actions

    @property
    def above(self):
        return self._above

    @above.setter
    def above(self, above):
        """
        The above step or policy will be set through this setter. Its
        observation and action spaces will match the this policy's interface.
        """
        assert self.interface[0] == above.observations
        assert self.interface[1] == above.actions
        self._above = above

    def begin_episode(self, training):
        self._assert_above()
        super().begin_episode(training)

    def step(self, observation):
        return super().step(observation)

    def _assert_above(self):
        if not self._above:
            raise RuntimeError('need to know above policy before simulation')


class Sequential(Step):

    def __init__(self, interface):
        super().__init__(interface)
        self.steps = []

    def add(self, step, *args, **kwargs):
        """
        Add a step or policy to the sequential behavior. If the step is just a
        policy, thus provides no interface, no further behavior can be added.
        """
        if inspect.isclass(step):
            step = step(self.interface, *args, **kwargs)
        elif args or kwargs:
            raise ValueError('unexpected args for an instantiated step')
        if self.steps:
            self.steps[-1].above = step
        if self.above:
            step.above = self.above
        self.steps.append(step)

    @property
    def interface(self):
        if not self.steps:
            return self.observations, self.actions
        self._assert_not_final()
        return self.steps[-1].interface

    @Step.above.setter
    def above(self, above):
        self._assert_not_final()
        # Workaround for super().above = above.
        super(Sequential, self.__class__).above.fset(self, above)
        if self.steps:
            self.steps[-1].above = above

    @property
    def final(self):
        return self.steps and not hasattr(self.steps[-1], 'interface')

    def begin_episode(self, training):
        super().begin_episode(training)
        for step in self.steps:
            step.begin_episode(training)

    def end_episode(self):
        super().end_episode()
        for step in self.steps:
            step.end_episode()

    def step(self, observation):
        super().step(observation)
        return self._first.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self._first.experience(*transition)

    @property
    def _first(self):
        if self.steps:
            return self.steps[0]
        else:
            self._assert_above()
            return self.above

    def _assert_above(self):
        if not self.final:
            super()._assert_above()

    def _assert_not_final(self):
        if self.final:
            raise RuntimeError('last step has no interface to add any further')
