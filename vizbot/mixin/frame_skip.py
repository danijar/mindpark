from vizbot.core import Mixin


class FrameSkip(Mixin):

    def __init__(self, agent, width=1):
        super().__init__(agent)
        self._width = width
        self._frames = None
        self._index = None
        self._action = self._agent._noop()

    def begin(self):
        self._agent.begin()
        self._frames = np.empty(self._states + (self._width,))
        self._index = 0

    def step(self, state):
        state = np.swapaxes(state, 0, -1)
        self._frames[self._index % self._width] = state
        self._index += 1
        if self._index and self._index % self._width:
            self._action = self._agent.step(self._frames.copy())
        return self._action
