from vizbot.core import Policy


class Skip(Policy):

    def __init__(self, interface, amount=4):
        super().__init__(interface)
        self._amount = amount
        self._action = None
        self._reward = None

    @property
    def interface(self):
        return self.observations, self.actions

    def begin_episode(self, training):
        super().begin_episode(training)
        self._reward = 0

    def observe(self, observation):
        super().observe(observation)
        # Show observations of frames 0, n, 2n, etc.
        if not self.step % self._amount:
            self._action = self.above.observe(observation)
        return self._action

    def receive(self, reward, final):
        super().receive(reward, final)
        # Collect all rewards that the current repeated action scored. Show
        # them at the end of the repeats, thus at frames n-1, 2n-1 etc.
        self._reward += reward
        if not (self.step + 1) % self._amount or final:
            self.above.receive(self._reward, final)
            self._reward = 0
