from test.fixtures import *


class TestTask:

    def test_shared_training(self, task, policy, env):
        steps = policy.recursive_steps
        assert all(x.task.training == task.training for x in steps)
        task.training = not task.training
        assert all(x.task.training == task.training for x in steps)

    def test_shared_epoch(self, task, policy, env):
        steps = policy.recursive_steps
        for epoch in range(task.epochs):
            value = task.epoch.increment()
            assert all(x.task.epoch == value for x in steps)
