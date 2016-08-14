from threading import Thread


class Simulator:

    def __init__(self, task, envs, policies):
        if len(envs) != len(policies):
            ValueError('must provide one policy for each env')
        self.task = task
        self.envs = envs
        self.policies = policies

    def __iter__(self):
        yield self._simulate(training=False)
        for epoch in range(self.task.epochs + 1):
            self.task.epoch += 1
            self._simulate(training=True)
            yield self._simulate(training=False)

    def _simulate(self, training):
        scores = []
        threads = []
        for env, policy in zip(self.envs, self.policies):
            args = env, policy, training, scores
            threads.append(Thread(target=self._worker, args=args))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return sum(scores) / len(scores)

    def _worker(self, env, policy, training, scores):
        if training:
            timestep = self.task.train_step
            timesteps = self.task.train_steps * self.task.epoch
        else:
            timestep = self.task.test_step
            timesteps = self.task.test_steps * (self.task.epoch + 1)
        while timestep < timesteps:
            score = self._episode(env, policy, training)
            scores.append(score)

    def _episode(self, env, policy, training):
        score = 0
        policy.begin_episode(training)
        observation = env.reset()
        while observation is not None:
            action = policy.step(observation)
            reward, successor = env.step(action)
            if training:
                policy.experience(observation, action, reward, successor)
                self.task.train_step += 1
            else:
                self.task.test_step += 1
            score += reward
            observation = successor
        policy.end_episode()
        return score
