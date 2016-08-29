from mindpark.utility import use_attrdicts


class Algorithm:

    @classmethod
    def defaults(self):
        discount = 0.95
        return locals()

    def __init__(self, task, config):
        self.task = task
        self.config = self._override_config(config)

    @property
    def policy(self):
        raise NotImplementedError

    @property
    def train_policies(self):
        return [self.policy]

    @property
    def test_policy(self):
        return self.policy

    def begin_epoch(self):
        pass

    def end_epoch(self):
        pass

    @classmethod
    def _override_config(cls, overrides):
        config = cls.defaults()
        for key in overrides:
            if key not in config:
                raise KeyError("unknown config key '{}'".format(key))
        config.update(overrides)
        config = use_attrdicts(config)
        return config
