from vizbot.core import Task


class BenchTask(Task):

    def __init__(self, train, test):
        self.current = train
        self.train = train
        self.test = test

    def set_training(self, training):
        if training:
            self.current = self.train
        else:
            self.current = self.test

    def __getattr__(self, key):
        return getattr(self.current, key)
