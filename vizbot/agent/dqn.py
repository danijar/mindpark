from pprint import pprint
import collections
import numpy as np
import tensorflow as tf
from vizbot.agent import EpsilonGreedy
from vizbot.model import Model, dense, default_network
from vizbot.preprocess import (
    Grayscale, Downsample, FrameSkip, NormalizeReward, NormalizeImage)
from vizbot.utility import Experience, Every, Statistic, merge_dicts


class DQN(EpsilonGreedy):

    @classmethod
    def defaults(cls):
        downsample = 2
        frame_skip = 4
        replay_capacity = int(2e4)
        batch_size = 32
        learning_rate = 1e-4
        optimizer = tf.train.RMSPropOptimizer
        epsilon_from = 1.0
        epsilon_to = 0.1
        epsilon_duration = 5e5
        print_cost = 10000
        start_learning = 100
        return merge_dicts(super().defaults(), locals())

    def __init__(self, trainer, config):
        # Preprocessing.
        trainer.add_preprocess(NormalizeReward)
        trainer.add_preprocess(NormalizeImage)
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, config.downsample)
        trainer.add_preprocess(FrameSkip, config.frame_skip)
        super().__init__(trainer, config)
        # Network.
        optimizer = (config.optimizer, float(config.learning_rate))
        self._actor = Model(self._create_network, optimizer)
        self._target = Model(self._create_network)
        self._target.weights = self._actor.weights
        print(str(self._actor))
        # Learning.
        self._memory = Experience(config.replay_capacity)
        self._costs = Statistic('Cost {:8.3f}', self.config.print_cost)

    def _create_network(self, model):
        state = model.add_input('state', self.states.shape)
        action = model.add_input('action', self.actions.shape)
        target = model.add_input('target')
        values = dense(default_network(state), self.actions.shape, tf.identity)
        model.add_output('value', tf.reduce_max(values, 1))
        model.add_output('choice',
            tf.one_hot(tf.argmax(values, 1), self.actions.shape))
        model.add_cost('cost',
            (tf.reduce_sum(action * values, 1) - target) ** 2)

    def _step(self, state):
        return self._actor.compute('choice', state=state)

    def _compute_target(self, reward, successor):
        finals = len(list(x for x in successor if x is None))
        if finals:
            print(finals, 'terminal states in current batch')
        future = self._target.compute('value', state=successor)
        final = np.isnan(successor.reshape((len(successor), -1))).any(1)
        future[final] = 0
        target = reward + self.config.discount * future
        return target

    def experience(self, state, action, reward, successor):
        self._memory.append((state, action, reward, successor))
        if len(self._memory) == 1:
            self._log_memory_size()
        if len(self._memory) < self.config.start_learning:
            return
        state, action, reward, successor = \
            self._memory.sample(self.config.batch_size)
        target = self._compute_target(reward, successor)
        self._target.weights = self._actor.weights
        cost = self._actor.train('cost', state=state, action=action, target=target)
        self._costs(cost)

    def _log_memory_size(self):
        size = self._memory.nbytes / (1024 ** 3)
        print('Replay memory size', round(size, 2), 'GB')
