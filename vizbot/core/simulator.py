from threading import Thread


class Simulator:

    """
    Process a task by simulating one or more policies on one environment each.
    If multiple policies are provided, they will be simluated in parallel.
    """

    def __init__(self, task, envs, policies, training):
        if len(envs) != len(policies):
            ValueError('must provide one policy for each env')
        if not all(task.interface == x.interface for x in envs):
            ValueError('envs must match the task interface')
        self._task = task
        self._envs = envs
        self._policies = policies
        self._training = training

    def __call__(self, amount):
        """
        Simulate approximately the specified amount of time steps of the task.
        Return the average score. If the task is already done, return None.
        """
        threads, scores = [], []
        target = min(self._task.step + amount, self._task.steps)
        for env, policy in zip(self._envs, self._policies):
            args = target, env, policy, scores
            threads.append(Thread(target=self._worker, args=args))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return sum(scores) / len(scores) if scores else None

    def _worker(self, target, env, policy, scores):
        while self._task.step < target:
            score = self._episode(env, policy)
            scores.append(score)

    def _episode(self, env, policy):
        score = 0
        policy.begin_episode(self._training)
        observation = env.reset()
        while observation is not None:
            action = policy.step(observation)
            reward, successor = env.step(action)
            if self._training:
                policy.experience(observation, action, reward, successor)
            self._task.step += 1
            score += reward
            observation = successor
        policy.end_episode()
        return score
