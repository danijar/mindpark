import functools
from threading import Thread
from vizbot.core import GymEnv, StopEpisode
from vizbot.utility import AttrDict, ensure_directory


class Trainer:

    def __init__(self, directory, env_name, agent_cls, agent_config,
                 epochs, train_steps, test_steps, videos=10):
        if directory:
            ensure_directory(directory)
        self._directory = directory
        self._env_name = env_name
        self._epochs = int(float(epochs))
        self._train_steps = int(float(train_steps))
        self._test_steps = int(float(test_steps))
        self._videos = directory and videos
        self._previous_step = 0
        self._train_step = 0
        self._test_step = 0
        self._preprocesses = []
        self._agent = agent_cls(self, AttrDict(agent_config))

    @property
    def directory(self):
        return self._directory

    @property
    def timesteps(self):
        return self._epochs * self._train_steps

    @property
    def timestep(self):
        return self._previous_step + self._train_step

    def add_preprocess(self, preprocess_cls, *args, **kwargs):
        self._preprocesses.append((preprocess_cls, args, kwargs))

    def create_env(self, monitoring=False):
        directory = monitoring and self._directory
        env = GymEnv(self._env_name, directory, self._video_callback)
        for preprocess, args, kwargs in self._preprocesses:
            env = preprocess(env, *args, **kwargs)
        return env

    def __iter__(self):
        for _ in range(self._epochs):
            self._agent.start_epoch()
            for learner in self._agent.learners:
                if learner is self._agent:
                    continue
                learner.start_epoch()
            self._train(self._agent.learners)
            score = self._test(self._agent.testee)
            for learner in self._agent.learners:
                if learner is self._agent:
                    continue
                learner.stop_epoch()
            self._agent.stop_epoch()
            yield score
        for learner in self._agent.learners:
            if learner is self._agent:
                continue
            learner.close()
        self._agent.close()

    def _train(self, learners):
        def target(env, learner):
            while self._train_step < self._train_steps:
                self._run_episode(env, learner, training=True)
            env.close()
        threads = []
        self._previous_step += self._train_step
        self._train_step = 0
        for learner in learners:
            assert len(learner.learners) == 1
            assert learner.learners[0] is learner
            env = self.create_env()
            threads.append(Thread(target=target, args=(env, learner)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def _test(self, testee):
        scores = []
        env = self.create_env(True)
        self._test_step = 0
        while self._test_step < self._test_steps:
            score = self._run_episode(env, testee, training=False)
            scores.append(score)
        env.close()
        return sum(scores) / len(scores)

    def _run_episode(self, env, agent, training):
        score = 0
        agent.start_episode(training)
        try:
            state = env.reset()
            # while self._test_step < self._test_steps:
            while True:
                action = agent.step(state)
                successor, reward = env.step(action)
                if training:
                    agent.experience(state, action, reward, successor)
                score += reward
                state = successor
                if training:
                    self._train_step += 1
                else:
                    self._test_step += 1
        except StopEpisode:
            agent.experience(state, action, reward, None)
        agent.stop_episode()
        return score

    def _video_callback(self, ignore):
        if not self._videos:
            return False
        return self._test_step % self._videos == 0
