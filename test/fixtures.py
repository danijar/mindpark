import pytest
from vizbot.core import Task
from test.mocks import Sequential, Identity, Skip, Random
from test.mocks import DurationEnv


@pytest.fixture(params=[1, 2, 17])
def duration(request):
    return request.param


@pytest.fixture
def env(duration):
    return DurationEnv(duration)


@pytest.fixture(params=[True, False])
def task(env, request):
    return Task(
        env.observs, env.actions, '/dev/null', env.duration, 3, request.param)


@pytest.fixture(params=range(9))
def policy(request, task):
    policy = Sequential(task)
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
        inner = Sequential(policy.above_task)
        inner.add(Skip, 2)
        inner.add(Identity)
        policy.add(inner)
        policy.add(Skip, 3)
    elif request.param == 5:
        # Nested.
        outer = policy
        for _ in range(5):
            inner = Sequential(policy.above_task)
            inner.add(Skip, 2)
            outer.add(inner)
            outer = inner
    elif request.param == 6:
        # Nested empty.
        outer = policy
        for _ in range(5):
            inner = Sequential(policy.above_task)
            outer.add(inner)
            outer = inner
    elif request.param == 7:
        # Concatenated.
        for _ in range(5):
            inner = Sequential(policy.above_task)
            inner.add(Skip, 2)
            policy.add(inner)
    elif request.param == 8:
        # Concatenated empty.
        for _ in range(5):
            inner = Sequential(policy.above_task)
            policy.add(inner)
    else:
        assert False
    policy.add(Random)
    steps = policy.recursive_steps
    print('Steps:', ', '.join([type(x).__name__ for x in steps]))
    return policy
