import pytest
from vizbot.core import Sequential, ExperienceProxy
from test.mocks import DurationEnv, Identity, Skip, Random


@pytest.fixture(params=[1, 2, 5, 21])
def duration(request):
    return request.param


@pytest.fixture
def env(duration):
    return DurationEnv(duration)


@pytest.fixture(params=[0, 1, 2])
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
        policy.begin_episode(False)
        reward, observation = None, env.reset()
        while observation is not None:
            for step in policy.steps:
                if step.timestep < 1:
                    continue
                print(step.timestep, step.transition)
                assert all(x is not None for x in step.transition)
            action = policy.observe(reward, observation)
            reward, observation = env.step(action)
        for step in policy.steps:
            if step.timestep < 1:
                continue
            assert all(x is not None for x in step.transition[:-1])
            # Terminal successor must be None.
            assert step.transition[-1] is None
        policy.end_episode()

    def test_observe_first_observation(self, env, policy):
        policy.begin_episode(False)
        reward, observation = None, env.reset()
        policy.observe(reward, observation)
        for level, step in enumerate(policy.steps):
            if level and step.timestep != 0:
                continue
            assert (step.observation == observation).all()
            assert step.reward is None
        policy.end_episode()

    def test_observe_last_reward(self, env, policy):
        policy.begin_episode(False)
        reward, observation = None, env.reset()
        while observation is not None:
            for step in policy.steps:
                step.reward = None
            action = policy.observe(reward, observation)
            reward, observation = env.step(action)
        for step in policy.steps:
            if step.timestep < 1:
                continue
            assert step.observation is None
            assert step.reward is not None
        policy.end_episode()

    def test_time_steps(self, env, policy):
        policy.begin_episode(False)
        reward, observation = None, env.reset()
        timestep = 0
        while observation is not None:
            for step in policy.steps:
                step.reward = None
            action = policy.observe(reward, observation)
            multiplier = 1
            for step in policy.steps:
                if step.timestep < 0:
                    break
                assert step.timestep == timestep // multiplier
                if isinstance(step, Skip):
                    multiplier *= step.amount
            reward, observation = env.step(action)
            timestep += 1
        policy.end_episode()
