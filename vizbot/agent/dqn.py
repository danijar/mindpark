import collections
import numpy as np
import tensorflow as tf
from vizbot.agent import EpsilonGreedy
from vizbot.model import Model, network_dqn
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, Experience


class DQN(EpsilonGreedy):

    @staticmethod
    def _config():
        discount = 0.99
        downsample = 4
        frame_skip = 4
        replay_capacity = int(1e5)
        batch_size = 32
        learning_rate = 1e-4
        optimizer = (tf.train.RMSPropOptimizer, 1e-3)
        epsilon = AttrDict(start=0.5, stop=0, over=int(5e5))
        return AttrDict(**locals())

    def __init__(self, trainer):
        self._config = self._config()
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, self._config.downsample)
        trainer.add_preprocess(FrameSkip, self._config.frame_skip)
        super().__init__(trainer, **self._config.epsilon)
        self._memory = Experience(self._config.replay_capacity)
        self._actor = Model(self._create_network, self._config.optimizer)
        self._target = Model(self._create_network)
        self._target.weights = self._actor.weights
        print(str(self._actor))
        self._costs = []

    def _step(self, state):
        return self._actor.compute('choice', state=state)

    def experience(self, state, action, reward, successor):
        self._memory.append((state, action, reward, successor))
        if len(self._memory) == 1:
            size = round(self._memory.nbytes / 1024 / 1024)
            print('Replay memory size', size, 'MB')
        if len(self._memory) < self._config.batch_size:
            return
        state, action, reward, successor = \
            self._memory.sample(self._config.batch_size)
        finals = len(list(x for x in successor if x is None))
        if finals:
            print(finals, 'terminal states in current batch')
        future = self._target.compute('value', state=successor)
        final = np.isnan(successor.reshape((len(successor), -1))).any(1)
        future[final] = 0
        assert np.isfinite(future).all()
        target = reward + self._config.discount * future
        self._target.weights = self._actor.weights
        cost = self._actor.train(
            'cost', state=state, action=action, target=target)
        assert np.isfinite(cost).all()
        self._costs.append(cost)

    def stop(self):
        if len(self._costs) < 2500:
            return
        print('Cost', sum(self._costs) / len(self._costs))
        self._costs = []

    def _create_network(self, model):
        state = model.add_input('state', self.states.shape)
        action = model.add_input('action', self.actions.shape)
        target = model.add_input('target')
        values = network_dqn(state, self.actions.shape)
        model.add_output('value', tf.reduce_max(values, 1))
        model.add_output('choice',
            tf.one_hot(tf.argmax(values, 1), self.actions.shape))
        model.add_cost('cost',
            (tf.reduce_sum(action * values, 1) - target) ** 2)
