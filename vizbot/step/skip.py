from vizbot.core import Policy


class Skip(Policy):

    # TODO: Have base class track both the global step and the step of the
    # current episode. How to name the properties?

    def __init__(self, interface, amount=4):
        super().__init__(interface)
        self._amount = amount
        self._action = None
        self._reward = None
        # For the modulo operation, we need the step of the current episode,
        # rather than the step of the overall simulation.
        self._step = None

    @property
    def interface(self):
        return self.observations, self.actions

    def begin_episode(self, training):
        super().begin_episode(training)
        self._reward = 0
        self._step = None

    def observe(self, observation):
        self._step = 0 if self._step is None else self._step + 1
        super().observe(observation)
        # Show observations of frames 0, n, 2n, etc.
        if not self._step % self._amount:
            self._action = self.above.observe(observation)
        return self._action

    def receive(self, reward, final):
        super().receive(reward, final)
        # Collect all rewards that the current repeated action scored. Show
        # them at the end of the repeats, thus at frames n-1, 2n-1 etc.
        self._reward += reward
        if not (self._step + 1) % self._amount or final:
            self.above.receive(self._reward, final)
            self._reward = 0
