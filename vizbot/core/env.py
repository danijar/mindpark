class Env:

    def __init__(self):
        self.episode = -1
        self.timestep = None
        self._agent = None

    @property
    def states(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError

    def register(self, agent):
        assert agent is not None
        if self._agent is not None:
            message = 'there is already agent {} registered to env {}'
            message = message.format(
                type(self._agent).__name__, type(self).__name__)
            raise RuntimeError(message)
        self._agent = agent

    def start(self):
        assert self.timestep is None
        self.episode += 1
        self.timestep = -1
        self._agent.start()

    def step(self):
        self.timestep += 1

    def stop(self):
        self._agent.stop()
        self.timestep = None
