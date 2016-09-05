import numpy as np
from mindpark.stats.figure import Figure
from mindpark.stats.scatter import Scatter
from mindpark.stats.reader import Metric
from mindpark.utility import natural_sorted


class Scores(Figure):

    """
    Generate figures showing algorithm comparisons on multiple environments as
    multiple plots.
    """

    def __init__(self):
        self._plot_score = Scatter(True, dict(alpha=0.5))

    def __call__(self, scores, title, filepath):
        self._validate_input(scores)
        fig, ax = self._create_subplots(2, len(scores))
        fig.suptitle(title, fontsize=16)
        scores = natural_sorted(scores.items(), key=lambda x: x[0])
        names, scores = zip(*scores)
        self._label_columns(ax, names)
        self._label_rows(ax, ['Training', 'Evaluation'])
        for index, algos in enumerate(scores):
            algos = {k: self._concat_runs(v) for k, v in algos.items()}
            ticks = {k: v.episode.max() for k, v in algos.items()}
            train = {k: v[v.training == 1] for k, v in algos.items()}
            test = {k: v[v.training == 0] for k, v in algos.items()}
            self._process_env(ax[0, index], train, ticks)
            self._process_env(ax[1, index], test, ticks)
        self._save(fig, filepath)

    def _process_env(self, ax, algos, ticks):
        algos = {k: v for k, v in algos.items() if len(v.step)}
        domains = {k: self._domain(v, ticks[k]) for k, v in algos.items()}
        lines = {k: v.data for k, v in algos.items()}
        self._plot_score(ax, domains, lines)

    def _concat_runs(self, runs):
        keys = sum([list(x.keys()) for x in runs], [])
        assert all((y in x for y in keys) for x in runs)
        return Metric({x: np.concatenate([y[x] for y in runs]) for x in keys})

    def _validate_input(self, scores):
        assert all(isinstance(x, dict) for x in scores.values())
        for algos in scores.values():
            for score in algos.values():
                assert isinstance(score, list)
                assert all(isinstance(x.data, np.ndarray) for x in score)
