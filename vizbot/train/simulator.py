from threading import Thread


class Simulator:

    def __init__(self, envs, policies, timesteps, training):
        self._envs = envs
        self._policies = policies
        self._timesteps = timesteps
        self._training = training
        self._timestep = None
        self._scores = None

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def timestep(self):
        return self._timestep

    def __call__(self):
        self._timestep = 0
        self._scores = []
        if not self._timesteps:
            return
        threads = []
        for env, policy in zip(self._envs, self._policies):
            threads.append(Thread(target=self._worker, args=(env, policy)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return sum(self._scores) / len(self._scores)

    def _worker(self, env, policy):
        while self._timestep < self._timesteps:
            score = self._episode(env, policy)
            self._scores.append(score)

    def _episode(self, env, policy):
        score = 0
        policy.begin_episode(self._training)
        observation = env.reset()
        # while self._test_step < self._test_steps:
        while observation is not None:
            action = policy.step(observation)
            reward, successor = env.step(action)
            self._timestep += 1
            score += reward
            if self._training:
                policy.experience(observation, action, reward, successor)
            observation = successor
        policy.end_episode()
        return score
