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
                 videos=0, experience=False, experience_maxlen=5e4):
        """
        Provide an interface to agents to train on an environment.

        Args:
            directory (path): Where to store results. Can be None for dry run.
            env_name (str): The name of a registered Gym environment.
            timesteps (int): The overall training duration in frames.
            videos (int): Every how many episodes to record a video. Zero
                disables recording.
            experience (bool): Whether to store transition tuples as Numpy
                arrays. Requires a lot of disk space.
            experience_maxlen (int): Maxmium amount of transitions to store per
                episode. Usually much higher than the normal episode length.
                Needed to allocate memory beforehand.
        """
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
        self._running = True

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
    def running(self):
        return self._running

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
        if not self._running:
            raise StopTraining(self)
        with self._get_env() as env:
            episode = self.episode
            score = self._run_episode(env, agent)
        self._scores.append(score)
        message = 'Episode {} timestep {} reward {}'
        print(message.format(episode, self.timestep, score))
        if self._timestep > self._timesteps:
            self._running = False
            self._close_envs()
            self._store_scores()

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
            env = GymEnv(self._env_name, self._directory, self._video_callback)
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

    def _video_callback(self, env_episode):
        if not self._videos:
            return False
        return self.episode % self._videos == 0
