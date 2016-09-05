import re
import os
import collections
from mindpark.stats.reader import Reader
from mindpark.stats.metrics import Metrics
from mindpark.stats.scores import Scores
from mindpark.utility import get_subdirs, read_yaml, natural_sorted


Run = collections.namedtuple(
    'Run', 'experiment name env algorithm repeat stats')


class Stats:

    """
    Core functionality of the stats sub command. Read metrics from an
    experiment using a Reader and plot the using Scores and Metrics figures.
    """

    def __init__(self, type_, selectors=None):
        self._type = type_
        self._read_scores = lambda x: next(Reader(['score'])(x))[1]
        self._plot_scores = Scores()
        self._read_metrics = Reader(selectors)
        self._plot_metrics = Metrics()

    def __call__(self, experiment):
        runs = self._collect_runs(experiment)
        self._create_scores_plot(runs)
        runs = [list(x.values()) for x in runs.values()]
        runs = sum(sum(runs, []), [])
        for run in runs:
            self._create_metrics_plot(run)

    def _create_scores_plot(self, envs):
        scores = {}
        for env, algos in envs.items():
            scores[env] = collections.defaultdict(list)
            for algo, runs in algos.items():
                scores[env][algo] = [self._read_scores(x.stats) for x in runs]
        name = os.path.basename(runs[0].experiment)
        title = re.findall(r'[A-Za-z]{2,}', name)
        title = ' '.join(x.title() for x in title)
        filepath = os.path.join(runs[0].experiment, name)
        self._plot_scores(scores, title, filepath)

    def _create_metrics_plot(self, run):
        title = '{} on {} (Repeat {})'.format(
            run.algorithm, run.env, run.repeat)
        print(' Plot run', title)
        metrics = list(self._read_metrics(run.stats))
        if not metrics:
            print('  No metrics found.')
            return
        filepath = '{}-{}-{}-{}.{}'.format(
            run.name, run.env, run.algorithm, run.repeat, self._type)
        filepath = os.path.join(run.experiment, filepath)
        self._plot_metrics(metrics, title, filepath)

    def _collect_runs(self, experiment):
        print('Read experiment', experiment)
        name = os.path.basename(experiment).title()
        runs = {}
        for env_dir in get_subdirs(experiment):
            env = os.path.basename(env_dir)
            runs[env] = collections.defaultdict(list)
            for directory in natural_sorted(get_subdirs(env_dir)):
                repeat = int(directory.rsplit('-', 1)[-1])
                algorithm = read_yaml(directory, 'algorithm.yaml').name
                stats = os.path.join(directory, 'stats.db')
                run = Run(experiment, name, env, algorithm, repeat, stats)
                runs[env][algorithm].append(run)
        return runs
