from gym.spaces import Box
from vizbot.core import Mixin


class Image(Mixin):

    def __init__(self, agent, color='color', subsample=1):
        super().__init__(agent)
        self._mode = color
        self._factor = subsample
        # TODO: Hide instead of over writing.
        shape = list(self._agent._states.shape)
        if self._mode == 'grayscale':
            shape = shape[: -1]
        shape[0] //= subsample
        shape[1] //= subsample
        self._agent._states = Box(0, 255, shape)

    def step(self, state):
        self._color(state, self._mode)
        state = self._subsample(state, self._factor)
        return self._agent.step(state)

    def _color(self, state, mode):
        if mode == 'color':
            return state
        elif mode == 'grayscale':
            return state.mean(-1)
        raise NotImplementedError

    def _subsample(self, state, factor):
        if not factor or factor == 1:
            return state
        if not isinstance(factor, int):
            raise NotImplementedError
        if len(state.shape) != 3:
            raise NotImplementedError
        (width, height), dtype = state.shape[:2], state.dtype
        shape = width // factor, factor, height // factor, factor, -1
        state = state.reshape(shape).mean(3).mean(1)
        state = state.astype(dtype)
        return state
