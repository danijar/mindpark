import pytest
from vizbot.core import Sequential
import vizbot.step
from test.mocks import DurationEnv, Random


STEPS = [
    'Identity', 'Maximum', 'Delta', 'Grayscale', 'Subsample', 'Skip',
    'History', 'Normalize', 'ClampReward', 'EpsilonGreedy']


@pytest.fixture(params=STEPS)
def step(request):
    return getattr(vizbot.step, request.param)


@pytest.fixture(params=(1, 2, 17))
def env(request):
    return DurationEnv(request.param)


@pytest.fixture
def interface(env):
    return env.interface


@pytest.fixture
def policy(interface, step):
    policy = Sequential(interface)
    print('Step:  ', step.__name__)
    print('Input: ', policy.interface[0])
    policy.add(step)
    print('Output:', policy.interface[0])
    policy.add(Random)
    return policy


class TestStep:

    def test_spaces(self, env, policy):
        # This tests the action and observation spaces since the mock env and
        # the mock monitored random policy check them.
        observation = env.reset()
        policy.begin_episode(True)
        while observation is not None:
            action = policy.observe(observation)
            reward, observation = env.step(action)
            policy.receive(reward, observation is None)
        policy.end_episode()
