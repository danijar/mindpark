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

    @property
    def train_policies(self):
        return self._train_policies

    @property
    def test_policy(self):
        return self._test_policy

    def begin_epoch(self, epoch):
        if self.epoch is None:
            assert epoch == 0
        else:
            assert epoch == self.epoch + 1
        self._epoch = epoch
        self._begin_epoch(epoch)

    def end_epoch(self):
        self._end_epoch()

    def close(self):
        self._close()

    @property
    def _train_policies(self):
        raise NotImplementedError

    @property
    def _test_policy(self):
        raise NotImplementedError

    def _begin_epoch(self, epoch):
        pass

    def _end_epoch(self):
        pass

    def _close(self):
        pass
