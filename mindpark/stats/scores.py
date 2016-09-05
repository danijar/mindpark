import numpy as np
import matplotlib.pyplot as plt
from mindpark.stats.scatter import Scatter


class Scores:

    """
    Generate figures showing algorithm comparisons on multiple environments as
    multiple plots.
    """

    def __init__(self):
        self._plot = Scatter()

    def __call__(self, scores, title, filepath):
        assert all(isinstance(x, dict) for x in scores.values())
        for algos in scores.values():
            for score in algos.values():
                assert isinstance(score, list)
                assert all(isinstance(x, np.ndarray) for x in score)
