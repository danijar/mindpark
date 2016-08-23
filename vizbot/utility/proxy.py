class Proxy:

    """
    Wrap objects and override some of their attributes. This allows to create
    changed versions of objects with the values of not overridden attributes
    staying in sync with the attributes of the original object. Overrides can
    be undone by deleting the attribute.
    """

    def __init__(self, inner):
        self.change(inner)

    def __setattr__(self, name, value):
        if not hasattr(self._inner, name):
            raise AttributeError
        super().__setattr__(name, value)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __delattr__(self, name):
        getattr(self, name)
        try:
            del self.__dict__[name]
        except KeyError:
            raise AttributeError

    def change(self, inner):
        super().__setattr__('_inner', inner)
