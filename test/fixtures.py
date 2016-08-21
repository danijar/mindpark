import pytest
from vizbot.core import Task
from test.mocks import DurationEnv


@pytest.fixture(params=[1, 2, 17])
def duration(request):
    return request.param


@pytest.fixture
def env(duration):
    return DurationEnv(duration)


@pytest.fixture
def task(env):
    return Task(env.observs, env.actions, '/dev/null', env.duration)
