class Env:

    def __init__(self):
        self.episode = None
        self._agent = None

    @property
    def states(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError

    def register(self, agent):
        if self._agent is not None:
            message = 'agent {} is already registered to env {}'
            message = message.format(
                type(self._agent).__name__, type(self).__name__)
            raise RuntimeError(message)
        assert self._agent is None
        assert agent is not None
        self._agent = agent

    def start(self):
        assert self.episode is None
        self.episode = 0
        self._agent.start()

    def step(self):
        self.episode += 1

    def stop(self):
        self.episode = None
        self._agent.stop()
