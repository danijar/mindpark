import pytest
import mindpark.algorithm
from mindpark.core import Task
from mindpark.utility import use_attrdicts
from test.mocks import Sequential, Identity, Skip, Random
from test.mocks import DurationEnv, MockViewer


@pytest.fixture(params=[1, 2, 17])
def duration(request):
    return request.param


@pytest.fixture
def env(duration):
    return DurationEnv(duration)


@pytest.fixture(params=[True, False])
def task(request, env, tmpdir):
    return Task(
        env.observs, env.actions, str(tmpdir), env.duration, 3, request.param)


@pytest.fixture(params=range(11))
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
        # Nested once.
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
    elif request.param == 9:
        # Prepended.
        policy.add(Skip, 2)
        outer = Sequential(policy.task)
        outer.add(Skip, 2)
        outer.add(policy)
        policy = outer
    elif request.param == 10:
        # Wrapped.
        policy.add(Skip, 2)
        outer = Sequential(policy.task)
        outer.add(Skip, 2)
        outer.add(policy)
        policy.add(Random)
        policy = outer
    else:
        assert False
    if policy.above_task:
        policy.add(Random)
    steps = policy.recursive_steps
    print('Steps:', ', '.join([type(x).__name__ for x in steps]))
    return policy


ALGOS = ['Random', 'DQN', 'A3C', 'KeyboardDoom', 'Reinforce']


@pytest.fixture(params=ALGOS)
def algo_cls(request):
    return getattr(mindpark.algorithm, request.param)


@pytest.fixture
def algo_config(algo_cls):
    config = use_attrdicts(algo_cls.defaults())
    if algo_cls.__name__ == 'DQN':
        config.replay_capacity = 100
        config.batch_size = 3
        config.start_learning = 10
        config.network = 'test'
        config.preprocess_config = dict(frame_skip=2)
    if algo_cls.__name__ == 'A3C':
        config.learners = 2
        config.network = 'test'
        config.preprocess_config = dict(frame_skip=2)
    if algo_cls.__name__ == 'KeyboardDoom':
        config.viewer = MockViewer
    if algo_cls.__name__ == 'Reinforce':
        config.update_every = 10
        config.batch_size = 5
        config.network = 'test'
    return config


@pytest.fixture
def algo(algo_cls, task, algo_config):
    return algo_cls(task, algo_config)
