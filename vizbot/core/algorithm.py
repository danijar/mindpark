from vizbot.utility import use_attrdicts


class Algorithm:

    @property
    @classmethod
    def defaults(self):
        discount = 0.95
        return locals()

    def __init__(self, observations, actions, config):
        self.observations = observations
        self.actions = actions
        self.config = use_attrdicts(config)
        self.epoch = None

    @property
    def train_policies(self):
        raise NotImplementedError

    @property
    def test_policy(self):
        raise NotImplementedError

    def begin_epoch(self, epoch):
        self.epoch = epoch

    def end_epoch(self):
        pass

    def close(self):
        pass
