import copy
import pytest
import vizbot.algorithm
from vizbot.core import Simulator
from vizbot.utility import use_attrdicts
from test.mocks import MockViewer
from test.fixtures import *


ALGOS = ['Random', 'DQN', 'A3C', 'KeyboardDoom']


@pytest.fixture(params=ALGOS)
def algo(request, task):
    algo_cls = getattr(vizbot.algorithm, request.param)
    config = use_attrdicts(algo_cls.defaults())
    if request.param == 'DQN':
        config.replay_capacity = 100
        config.batch_size = 5
        config.start_learning = 10
        config.network = 'network_test'
    if request.param == 'A3C':
        config.learners = 2
        config.network = 'network_test'
    if request.param == 'KeyboardDoom':
        config.viewer = MockViewer
    return algo_cls(task, config)


class TestAlgorithm:

    def test_no_error(self, env, task, algo):
        policies = algo.train_policies if task.training else [algo.test_policy]
        envs = [copy.deepcopy(env) for _ in policies]
        simulator = Simulator(task, policies, envs)
        for score in simulator:
            assert score is not None
