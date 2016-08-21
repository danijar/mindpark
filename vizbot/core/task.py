from vizbot.utility import Counter


class Task:

    """
    A specification of a task that can be simulated. Will be provided to the
    algorithm to access the environment interface, the current time step, the
    maximum time steps, and a directory to store results like checkpoints in.
    """

    def __init__(
            self, observs, actions, directory, steps, step=None, episode=None):
        self.observs = observs
        self.actions = actions
        self.steps = steps
        self.directory = directory
        self.step = step or Counter()
        self.episode = episode or Counter()
