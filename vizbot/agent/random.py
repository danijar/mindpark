from vizbot.core import Agent


class Random(Agent):

    def step(self, state):
        super().step(state)
        action = self._env.action_space.sample()
        action = self._fix_doom_deatchmatch(action)
        return action

    def _fix_doom_deatchmatch(self, action):
        if self._env.spec.id == 'DoomDeathmatch-v0':
            action[33] = 0
        return action
