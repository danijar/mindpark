from mindpark.core import Metric
from mindpark.step import Identity


class Score(Identity):

    """
    Store the score at the end of each episode. Does not alter behavior.
    """

    def __init__(self, task):
        super().__init__(task)
        self._score_metric = Metric(self.task, 'score', 1)
        self._score = 0

    def receive(self, reward, final):
        self._score += reward
        if final:
            self._score_metric(self._score)
            self._score = 0
        super().receive(reward, final)
