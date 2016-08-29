from mindpark.utility import Counter


class Task:

    """
    A specification of a task that can be simulated. Will be provided to the
    algorithm to access the environment interface, the current time step, the
    maximum time steps, and a directory to store results like checkpoints in.
    """

    def __init__(
            self, observs, actions, directory, steps, epochs, training,
            step=None, epoch=None, episode=None):
        required = (observs, actions, steps, epochs, training)
        assert all(x is not None for x in required)
        self.observs = observs
        self.actions = actions
        self.directory = directory
        self.steps = steps
        self.epochs = epochs
        self.training = training
        self.step = step or Counter()
        self.epoch = epoch or Counter()
        self.episode = episode or Counter()
