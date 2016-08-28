import copy
import pytest
import vizbot.algorithm
from vizbot.core import Simulator
from vizbot.utility import use_attrdicts
from test.mocks import MockViewer
from test.fixtures import *


ALGOS = ['Random', 'DQN', 'A3C', 'KeyboardDoom']


@pytest.fixture(params=ALGOS)
def algo_cls(request):
    return getattr(vizbot.algorithm, request.param)


@pytest.fixture
def algo(algo_cls, task):
    config = use_attrdicts(algo_cls.defaults())
    if algo_cls.__name__ == 'DQN':
        config.replay_capacity = 100
        config.batch_size = 5
        config.start_learning = 10
        config.network = 'network_test'
    if algo_cls.__name__ == 'A3C':
        config.learners = 2
        config.network = 'network_test'
    if algo_cls.__name__ == 'KeyboardDoom':
        config.viewer = MockViewer
    return algo_cls(task, config)


class TestAlgorithm:

    def test_no_error(self, env, task, algo):
        policies = algo.train_policies if task.training else [algo.test_policy]
        envs = [copy.deepcopy(env) for _ in policies]
        simulator = Simulator(task, policies, envs)
        for score in simulator:
            assert score is not None
