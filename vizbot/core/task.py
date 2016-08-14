import gym


class Task:

    def __init__(self, directory, env_name, epochs, test_steps, train_steps):
        self.directory = directory
        self.env_name = env_name
        self.interface = self._determine_interface(env_name)
        self.epochs = epochs
        self.epoch = 0
        self.train_steps = train_steps
        self.train_steps = 0
        self.test_steps = test_steps
        self.test_step = 0

    @property
    def timesteps(self):
        return self.epochs * self.train_steps

    @property
    def progress(self):
        return self.timestep / self.timesteps

    @staticmethod
    def _determine_interface(env_name):
        env = gym.make(env_name)
        interface = env.observation_space, env.action_space
        env.close()
        return interface
