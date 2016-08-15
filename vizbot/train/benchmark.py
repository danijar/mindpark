import itertools
import os
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from vizbot.train.definition import Definition
from vizbot.train.job import Job


class Benchmark:

    """
    Train each algorithm on each environment for multiple repeats and store
    statistics and recordings in the experiment directory.
    """

    def __init__(self, directory=None, parallel=1, videos=False,
                 stacktraces=True):
        if directory:
            directory = os.path.abspath(os.path.expanduser(directory))
        self._directory = directory
        self._parallel = parallel
        self._videos = videos
        self._stacktraces = stacktraces
        self._lock = Lock()

    def __call__(self, definition):
        start = time.time()
        definition = Definition(definition)
        experiment = self._start_experiment(definition)
        if experiment:
            name = os.path.basename(experiment)
            self._dump_yaml(self._definition, experiment, name + '.yaml')
        tasks = itertools.product(
            range(self._definition.repeats),
            self._definition.envs,
            self._definition.algorithms)
        with ThreadPoolExecutor(max_workers=self._parallel) as executor:
            for repeat, env_name, algo_conf in tasks:
                job = Job(experiment, env_name, algo_conf, repeat, definition)
                executor.submit(job.__call__)
        message = 'Congratulations, benchmark finished after {} hours'
        duration = round((time.time() - start) / 3600, 1)
        self._print_headline(message.format(duration), style='=')
        if experiment:
            print('Find results in', experiment)

    def _start_experiment(self):
        self._print_headline('Start experiment', style='=')
        if not self._directory:
            print('Dry run; no results will be stored!')
            return None
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
        name = '{}-{}'.format(timestamp, self._definition.experiment)
        experiment = os.path.join(self._directory, name)
        print('Result will be stored in', experiment)
        return experiment
