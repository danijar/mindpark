import pytest
import vizbot.step
from vizbot.core import Sequential
from test.mocks import Random
from test.fixtures import *


STEPS = [
    'Identity', 'Maximum', 'Delta', 'Grayscale', 'Subsample', 'Skip',
    'History', 'Normalize', 'ClampReward', 'EpsilonGreedy']


@pytest.fixture(params=STEPS)
def step(request):
    return getattr(vizbot.step, request.param)


@pytest.fixture
def policy(task, step):
    policy = Sequential(task)
    print('Step:  ', step.__name__)
    print('Input: ', policy.task.observs)
    policy.add(step)
    print('Output:', policy.task.actions)
    policy.add(Random)
    return policy


class TestStep:

    def test_spaces(self, env, policy):
        # This tests the action and observ spaces since the mock env and
        # the mock monitored random policy check them.
        observ = env.reset()
        policy.begin_episode(0, True)
        while observ is not None:
            action = policy.observe(observ)
            reward, observ = env.step(action)
            policy.receive(reward, observ is None)
        policy.end_episode()
