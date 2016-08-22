from vizbot.core import Partial


class RandomStart(Partial):

    def __init__(self, task, max_noop=30):
        super().__init__(task)
        self._max_noop = max_noop
        self._noop = 0
        self._noops = None

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return self.task.actions

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._noops = self.random.randint(0, self._max_noop)

    def observe(self, observ):
        super().observe(observ)
        if self._noops:
            self._noops -= 1
            return self._noop
        self._noops = None
        return self.above.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        if self._noops is None:
            self.above.receive(reward, final)
