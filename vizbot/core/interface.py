class Interface:

    @property
    def observations(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError
