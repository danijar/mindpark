from mindpark.core import Simulator
from test.fixtures import *


second_policy = policy
second_env = env


class TestSimulator:

    def test_individual_episode_when_parallel(
            self, task, policy, second_policy, env, second_env):
        def equal(x):
            x = list(x)
            return all(y == x[0] for y in x)
        policies = [policy, second_policy]
        envs = [env, second_env]
        simulator = Simulator(task, policies, envs)
        for _ in range(task.epochs):
            simulator()
            assert equal(x.task.episode for x in policies)
