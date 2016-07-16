class Add:

    def __init__(self, cls, *args, **kwargs):
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self, wrapped_cls):
        class Wrapped(self._cls):
            def __new__(cls, states, actions, *args, **kwargs):
                def construct(mixin_states, mixin_actions):
                    return wrapped_cls(
                        mixin_states, mixin_actions, *args, **kwargs)
                return self._cls(
                    construct, states, actions, *self._args, **self._kwargs)
        return Wrapped


class Mixin:

    def __init__(self, agent_cls):
        self._agent = agent_cls(self.states, self.actions)

    @property
    def states(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(self._agent, name)
