import mindpark as mp


class Configurable:

    def __init__(self, config):
        self.config = self._override_config(config)

    @classmethod
    def defaults(self):
        discount = 0.95
        return locals()

    @classmethod
    def _override_config(cls, overrides):
        config = cls.defaults()
        for key in overrides:
            if key not in config:
                raise KeyError("unknown config key '{}'".format(key))
        config.update(overrides)
        config = mp.utility.use_attrdicts(config)
        return config
