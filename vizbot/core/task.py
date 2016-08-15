class Task:

    """
    A specification of a task that can be simulated. Will be provided to the
    algorithm to access the environment interface, the current time step, the
    maximum time steps, and a directory to store results like checkpoints in.
    """

    def __init__(self, interface, steps, directory):
        self.interface = interface
        self.steps = steps
        self.directory = directory
        self.step = 0
