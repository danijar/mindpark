from vizbot.core import Algorithm
from vizbot.step import Random as Policy


class Random(Algorithm):

    def __init__(self, interface, config):
        super().__init__(interface, config)
        self._policy = Policy(self.interface)

    @property
    def train_policies(self):
        return [self._policy]

    @property
    def test_policy(self):
        return self._policy
