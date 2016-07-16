class Env:

    def __init__(self):
        self._actor = None

    @property
    def states(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError

    def register(self, actor):
        if self._actor:
            raise ValueError
        self._actor = actor

    def begin(self):
        self._actor.begin()

    def end(self):
        self._actor.end()

    def step(self, state):
        return self._actor.step(state)


class Preprocess:

    pass


class MyAgent(Agent):

    def __init__(self, env):
        env = MyEnv(env, params)
        env = MyEnv(env, params)
        super().__init__(self, env)
