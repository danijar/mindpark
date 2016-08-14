from vizbot.utility import use_attrdicts


class Algorithm:

    @classmethod
    def defaults(self):
        discount = 0.95
        return locals()

    def __init__(self, task, config):
        self.task = task
        self.config = use_attrdicts(config)
        self.epoch = None

    @property
    def policy(self):
        raise NotImplementedError

    @property
    def train_policies(self):
        return [self.policy]

    @property
    def test_policy(self):
        return self.policy

    def begin_epoch(self, epoch):
        self.epoch = epoch

    def end_epoch(self):
        pass
