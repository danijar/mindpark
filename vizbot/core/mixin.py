from vizbot.core.agent import Agent


class Mixin:
    """
    Base class for features that can be wrapped around agents. Mixins can
    overwrite any agent properties. Multiple mixins wrap around each other.
    """

    def __init__(self, agent):
        self._agent = agent

    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], Agent):
            self = super().__new__(cls)
            # self.__init__(*args, **kwargs)
        else:
            self = object.__new__(Mixin)
            self._cls = cls
            self._args = args
            self._kwargs = kwargs
        return self

    def __call__(self, agent_cls):
        # class Wrapped(agent_cls):
        #     def __new__(cls, *args, **kwargs):
        #         obj = super().__new__(cls)
        #         obj = self._cls(obj, *self._args, **self._kwargs)
        #         return obj
        # return Wrapped
        agent_cls._original_new = agent_cls.__new__
        def new(cls, *args, **kwargs):
            obj = object.__new__(cls)
            obj.__init__(*args, **kwargs)
            obj = self._cls(obj, *self._args, **self._kwargs)
            return obj
        agent_cls.__new__ = new
        return agent_cls

    def __getattr__(self, name):
        return getattr(self._agent, name)
