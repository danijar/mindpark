import collections
import numpy as np
import tensorflow as tf
from vizbot.core import Model
from vizbot.agent import EpsilonGreedy
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, Experience, lazy_property
from vizbot.utility import dense, conv2d


class DQN(EpsilonGreedy):

    @staticmethod
    def _config():
        discount = 0.99
        downsample = 4
        frame_skip = 4
        replay_capacity = int(1e5)
        batch_size = 32
        learning_rate = 1e-4
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        epsilon = AttrDict(start=0.5, stop=0, over=int(5e5))
        return AttrDict(**locals())

    def __init__(self, trainer):
        self._config = self._config()
        super().__init__(trainer, **self._config.epsilon)
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, self._config.downsample)
        trainer.add_preprocess(FrameSkip, self._config.frame_skip)
        self._memory = Experience(self._config.replay_capacity)
        self._actor = Model(self._build_network, self._config.optimizer)
        self._target = Model(self._build_network)
        self._target.weights = self._actor.weights
        print('States', self.states.shape)
        print(str(self._actor))

    def _step(self, state):
        return self._actor.compute('act', state=state)

    def start(self):
        self._costs = []

    def experience(self, state, action, reward, successor):
        if reward > 0:
            print(['DQN', 'Greedy'][self._was_greedy], 'reward', reward)
        self._memory.append((state, action, reward, successor))
        if len(self._memory) == 1:
            size = round(self._memory.nbytes / 1024 / 1024)
            print('Replay memory size', size, 'MB')
        if len(self._memory) < self._config.batch_size:
            return
        state, action, reward, successor = \
            self._memory.sample(self._config.batch_size)
        # print('Final states in batch:',
        #       len(list(x for x in successor if x is None)))
        future = self._target.compute('best', state=successor)
        final = np.isnan(successor.reshape((len(successor), -1))).any(1)
        future[final] = 0
        target = reward + self._config.discount * future
        self._target.weights = self._actor.weights
        costs = self._actor.train(
            'cost', state=state, action=action, target=target)
        self._costs += costs

    def stop(self):
        print('DQN cost', sum(self._costs) / len(self._costs))

    def _build_network(self, model):
        state = model.add_input('state', self.states.shape)
        action = model.add_input('action', self.actions.shape)
        target = model.add_input('target')
        x = conv2d(state, 16, 4, 3, tf.nn.elu, 2)
        x = conv2d(x, 32, 2, 1, tf.nn.elu)
        x = dense(x, 256, tf.nn.elu)
        x = dense(x, 256, tf.nn.elu)
        x = dense(x, self.actions.shape, tf.nn.elu)
        model.add_output('best', tf.reduce_max(x, 1))
        model.add_output('act', tf.one_hot(tf.argmax(x, 1), self.actions.shape))
        model.add_cost('cost', (tf.reduce_sum(action * x, 1) - target) ** 2)
