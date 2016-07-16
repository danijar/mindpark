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
        assert self._agent is None
        assert agent is not None
        self._agent = agent

    def begin(self):
        assert self.episode is None
        self.episode = 0
        self._agent.begin()

    def step(self):
        self.episode += 1

    def end(self):
        self.episode = None
        self._agent.end()
