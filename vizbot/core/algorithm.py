from vizbot.utility import use_attrdicts


class Algorithm:

    @property
    @classmethod
    def defaults(self):
        discount = 0.95
        return locals()

    def __init__(self, observations, actions, config):
        self._observations = observations
        self._actions = actions
        self._config = use_attrdicts(config)
        self._epoch = None

    @property
    def config(self):
        return self._config

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def epoch(self):
        if self._epoch is None:
            raise RuntimeError('no episode started')
        return self._epoch

    def begin_epoch(self, epoch):
        self._epoch = epoch

    def end_epoch(self):
        pass

    @property
    def train_policies(self):
        raise NotImplementedError

    @property
    def test_policy(self):
        raise NotImplementedError
