import os
import contextlib
import queue
import numpy as np
from vizbot.core import GymEnv, StopEpisode
from vizbot.utility import Experience, ensure_directory


class StopTraining(Exception):

    def __init__(self, trainer):
        super().__init__(self)
        self.trainer = trainer


class Trainer:

    def __init__(self, directory, env_name, timesteps,
                 videos=False, experience=False, experience_maxlen=1e6):
        if directory:
            ensure_directory(directory)
        self._directory = directory
        self._env_name = env_name
        self._timesteps = timesteps
        self._videos = videos
        self._experience = experience
        self._experience_maxlen = experience_maxlen
        self._envs = queue.Queue()
        self._preprocesses = []
        self._scores = []
        self._episode = 0
        self._timestep = 0
        self._states = None
        self._actions = None

    @property
    def states(self):
        if self._states is None:
            with self._get_env() as env:
                self._states = env.states
                self._actions = env.actions
        return self._states

    @property
    def actions(self):
        if self._actions is None:
            with self._get_env() as env:
                self._states = env.states
                self._actions = env.actions
        return self._actions

    @property
    def episode(self):
        return self._episode

    @property
    def timestep(self):
        return self._timestep

    @property
    def scores(self):
        return self._scores

    def add_preprocess(self, preprocess_cls, *args, **kwargs):
        if not self._envs.empty():
            print('Warning: Close existing envs to add preprocess.')
            self._close_envs()
            self._states = None
            self._actions = None
        self._preprocesses.append((preprocess_cls, args, kwargs))

    def run_episode(self, agent):
        if self._timestep > self._timesteps:
            self._close_envs()
            self._store_scores()
            raise StopTraining(self)
        with self._get_env() as env:
            episode = self.episode
            score = self._run_episode(env, agent)
        self._scores.append(score)
        message = 'Episode {} timestep {} reward {}'
        print(message.format(episode, self.timestep, score))

    def _run_episode(self, env, agent):
        episode = self._episode
        self._episode += 1
        if self._directory and self._experience:
            experience = Experience(self._experience_maxlen)
        score = 0
        agent.start()
        try:
            state = env.reset()
            while True:
                action = agent.step(state)
                successor, reward = env.step(action)
                transition = (state, action, reward, successor)
                agent.experience(*transition)
                if self._directory and self._experience:
                    experience.append(transition)
                state = successor
                score += reward
                self._timestep += 1
        except StopEpisode:
            pass
        agent.stop()
        if self._directory and self._experience:
            experience.save(os.path.join(
                self._directory, 'experience-{}.npz'.format(episode)))
        return score

    @contextlib.contextmanager
    def _get_env(self):
        try:
            env = self._envs.get(timeout=0.1)
        except queue.Empty:
            env = GymEnv(self._env_name, self._directory, self._videos)
            for preprocess, args, kwargs in self._preprocesses:
                env = preprocess(env, *args, **kwargs)
        yield env
        self._envs.put(env)

    def _close_envs(self):
        for env in iter(self._env.get, 'STOP'):
            env.close()

    def _store_scores(self):
        if self._directory:
            scores = np.array(self._scores)
            np.save(os.path.join(self._directory, 'scores.npy'), scores)
