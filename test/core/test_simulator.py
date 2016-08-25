import copy
from vizbot.core import Simulator
from test.fixtures import *


class TestSimulator:

    def test_individual_episode_when_parallel(self, task, policy, env):
        def equal(x):
            x = list(x)
            return all(y == x[0] for y in x)
        policies = copy.copy(policy), copy.copy(policy)
        envs = copy.copy(env), copy.copy(env)
        simulator = Simulator(task, policies, envs)
        for _ in range(task.epochs):
            simulator()
            assert equal(x.task.episode for x in policies)
