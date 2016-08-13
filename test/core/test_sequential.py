import pytest
from test.mocks import DurationEnv, Sequential, Identity, Skip, Random


@pytest.fixture(params=[1, 2, 5, 21])
def duration(request):
    return request.param


@pytest.fixture
def env(duration):
    return DurationEnv(duration)


@pytest.fixture
def interface(env):
    return env.interface


@pytest.fixture(params=range(9))
def policy(request, interface):
    policy = Sequential(interface)
    if request.param == 0:
        # Empty.
        pass
    elif request.param == 1:
        # Sequential.
        policy.add(Identity)
        policy.add(Identity)
    elif request.param == 2:
        # Timescales.
        policy.add(Skip, 2)
    elif request.param == 3:
        # Interleaved.
        policy.add(Identity)
        policy.add(Skip, 2)
        policy.add(Identity)
        policy.add(Skip, 3)
        policy.add(Identity)
    elif request.param == 4:
        # Nested ones.
        policy.add(Skip, 2)
        inner = Sequential(policy.interface)
        inner.add(Skip, 2)
        inner.add(Identity)
        policy.add(inner)
        policy.add(Skip, 3)
    elif request.param == 5:
        # Nested.
        outer = policy
        for _ in range(5):
            inner = Sequential(policy.interface)
            inner.add(Skip, 2)
            outer.add(inner)
            outer = inner
    elif request.param == 6:
        # Nested empty.
        outer = policy
        for _ in range(5):
            inner = Sequential(policy.interface)
            outer.add(inner)
            outer = inner
    elif request.param == 7:
        # Concatenated.
        for _ in range(5):
            inner = Sequential(policy.interface)
            inner.add(Skip, 2)
            policy.add(inner)
    elif request.param == 8:
        # Concatenated empty.
        for _ in range(5):
            inner = Sequential(policy.interface)
            policy.add(inner)
    else:
        assert False
    policy.add(Random)
    print('Steps:', ', '.join([type(x).__name__ for x in policy.steps]))
    return policy


class TestSequential:

    def test_add_type_args(self, interface):
        policy = Sequential(interface)
        policy.add(Skip, 42)
        assert policy.steps[-1]._amount == 42
        policy.add(Skip, amount=13)
        assert policy.steps[-1]._amount == 13

    def test_need_interface_to_add(self, interface):
        policy = Sequential(interface)
        policy.add(Skip, 42)
        policy.add(Random)
        with pytest.raises(RuntimeError):
            policy.add(Identity)

    def test_experience_is_not_none(self, env, policy):
        policy.begin_episode(True)
        observation = env.reset()
        while observation is not None:
            for step in policy.steps:
                if not step.timestep:
                    continue
                assert all(x is not None for x in step.transition)
            action = policy.step(observation)
            reward, successor = env.step(action)
            policy.experience(observation, action, reward, successor)
            observation = successor
        policy.end_episode()

    def test_experience_last_transition(self, env, policy):
        policy.begin_episode(True)
        observation = env.reset()
        while observation is not None:
            action = policy.step(observation)
            reward, successor = env.step(action)
            for step in policy.steps:
                step.transition = None
            policy.experience(observation, action, reward, successor)
            observation = successor
        for step in policy.steps:
            if step.timestep is None:
                break
            assert all(x is not None for x in step.transition[:-1])
            assert step.transition[-1] is None
        policy.end_episode()

    def test_time_steps(self, env, policy):
        policy.begin_episode(True)
        timestep = 0
        observation = env.reset()
        while observation is not None:
            action = policy.step(observation)
            reward, successor = env.step(action)
            policy.experience(observation, action, reward, successor)
            observation = successor
            timesteps, computeds = [], []
            steps, computed = policy.steps[:], timestep
            while steps:
                step = steps.pop(0)
                if step.timestep is None:
                    break
                timesteps.append(step.timestep)
                computeds.append(computed)
                if hasattr(step, '_amount'):
                    computed = computed // step._amount
                if hasattr(step, 'steps'):
                    steps = step.steps + steps
            assert timesteps == computeds
            timestep += 1
        policy.end_episode()
