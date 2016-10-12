from mindpark.utility import Configurable


class Algorithm(Configurable):

    @classmethod
    def defaults(self):
        discount = 0.95
        return locals()

    def __init__(self, task, config):
        super().__init__(config)
        self.task = task

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
