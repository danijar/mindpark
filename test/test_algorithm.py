import copy
import pytest
from mindpark.core import Simulator
from test.fixtures import *


class TestAlgorithm:

    def test_no_error(self, env, task, algo):
        policies = algo.train_policies if task.training else [algo.test_policy]
        envs = [copy.deepcopy(env) for _ in policies]
        simulator = Simulator(task, policies, envs)
        for score in simulator:
            assert score is not None

    def test_unknown_config_key(self, task, algo_cls):
        config = dict(foo=42)
        with pytest.raises(KeyError):
            algo_cls(task, config)
