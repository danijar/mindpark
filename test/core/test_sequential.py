import pytest
from vizbot.core import Sequential
from test.mocks import DurationEnv, Identity, Skip, Random


@pytest.fixture(params=[1, 2, 5, 21])
def duration(request):
    return request.param


@pytest.fixture
def env(duration):
    return DurationEnv(duration)


@pytest.fixture(params=[0, 1, 2, 3])
def policy(request, env):
    policy = Sequential(env.observations, env.actions)
    if request.param == 0:
        pass
    elif request.param == 1:
        policy.add(Identity)
        policy.add(Identity)
    elif request.param == 2:
        policy.add(Skip, 2)
    elif request.param == 3:
        policy.add(Identity)
        policy.add(Skip, 2)
        policy.add(Identity)
        policy.add(Skip, 3)
        policy.add(Identity)
    else:
        assert False
    policy.add(Random)
    return policy


class TestSequential:

    def test_add_type_args(self, env):
        policy = Sequential(env.observations, env.actions)
        policy.add(Skip, 42)
        assert policy.steps[-1].amount == 42
        policy.add(Skip, amount=13)
        assert policy.steps[-1].amount == 13

    def test_experience_is_not_none(self, env, policy):
        policy.begin_episode(True)
        reward, observation = None, env.reset()
        while True:
            for step in policy.steps:
                if not step.timestep:
                    continue
                assert all(x is not None for x in step.transition)
            action = policy.observe(reward, observation)
            if observation is None:
                break
            reward, observation = env.step(action)
        assert observation is None
        for level, step in enumerate(policy.steps):
            if step.timestep is None:
                continue
            assert all(x is not None for x in step.transition[:-1])
            # Terminal successor must be None.
            assert step.transition[-1] is None
        policy.end_episode()

    def test_observe_first_observation(self, env, policy):
        policy.begin_episode(True)
        reward, observation = None, env.reset()
        policy.observe(reward, observation)
        for level, step in enumerate(policy.steps):
            if level and step.timestep != 0:
                continue
            assert (step.observation == observation).all()
            assert step.reward is None
        policy.end_episode()

    def test_observe_last_reward(self, env, policy):
        policy.begin_episode(True)
        reward, observation = None, env.reset()
        while True:
            for step in policy.steps:
                step._policy.reward = None
            action = policy.observe(reward, observation)
            if observation is None:
                break
            reward, observation = env.step(action)
        assert observation is None
        assert reward is not None
        for level, step in enumerate(policy.steps):
            if step.timestep is None:
                continue
            assert step.observation is None
            assert step.reward is not None
        policy.end_episode()

    @pytest.skip()
    def test_time_steps(self, env, policy):
        import math
        policy.begin_episode(True)
        reward, observation = None, env.reset()
        timestep = 0
        while True:
            for step in policy.steps:
                step._policy.reward = None
            action = policy.observe(reward, observation)
            local = timestep
            for level, step in enumerate(policy.steps):
                if step.timestep is None:
                    break
                assert step.timestep == local
                if hasattr(step, 'amount'):
                    local = local // step.amount
            if observation is None:
                break
            reward, observation = env.step(action)
            timestep += 1
        policy.end_episode()
