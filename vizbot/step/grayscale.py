from vizbot.step.filter import Filter


class Grayscale(Filter):

    """
    Convert observations to grayscale, dropping their last dimension. The
    default weighting of the RGB channels extracts the luminance.
    """

    def __init__(self, task, weighting=(0.299, 0.587, 0.114)):
        super().__init__(task)
        if len(weighting) != self.task.observs.shape[-1]:
            raise ValueError('weighting must match last axis of observations')
        self._weighting = weighting

    def filter(self, observ):
        return (self._weighting * observ).sum(-1)
