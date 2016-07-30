import numpy as np
import tensorflow as tf
from vizbot.agent import EpsilonGreedy
from vizbot import model as networks
from vizbot.model import Model, dense
from vizbot.preprocess import (
    Grayscale, Downsample, FrameSkip, NormalizeReward, NormalizeImage)
from vizbot.utility import Experience, Decay, Every, merge_dicts


class DQN(EpsilonGreedy):

    @classmethod
    def defaults(cls):
        # Preprocesses.
        downsample = 2
        frame_skip = 7
        delta = True
        # Exploration.
        epsilon_from = 1.0
        epsilon_to = 0.1
        epsilon_duration = 1e6
        test_epsilon = 0.05
        # Learning.
        network = 'network_dqn'
        replay_capacity = int(1e5)  # 1e6
        epsilon_after = replay_capacity
        batch_size = 32
        initial_learning_rate = 1e-4
        optimizer = tf.train.RMSPropOptimizer
        rms_decay = 0.99
        sync_target = 32
        start_learning = replay_capacity
        return merge_dicts(super().defaults(), locals())

    def __init__(self, trainer, config):
        self._add_preprocesses(trainer, config)
        super().__init__(trainer, config)
        # Network.
        self._actor = Model(self._create_network)
        self._target = Model(self._create_network)
        self._target.weights = self._actor.weights
        self._sync_target = Every(config.sync_target)
        # print(str(self._actor))
        # Learning.
        self._memory = Experience(config.replay_capacity)
        self._learning_rate = Decay(
            float(config.initial_learning_rate), 0, self._trainer.timesteps)
        self._costs = None
        self._maxqs = None

    def start_epoch(self):
        super().start_epoch()
        self._costs = []
        self._maxqs = []

    def stop_epoch(self):
        super().stop_epoch()
        if self._costs:
            print('Cost {:8.3f}'.format(sum(self._costs) / len(self._costs)))
        if self._maxqs:
            print('MaxQ {:8.3f}'.format(sum(self._maxqs) / len(self._maxqs)))
        if self._trainer.directory:
            self._actor.save(self._trainer.directory, 'model')

    def experience(self, state, action, reward, successor):
        self._memory.append((state, action, reward, successor))
        if len(self._memory) == 1:
            self._log_memory_size()
        if len(self._memory) < self.config.start_learning:
            return
        state, action, reward, successor = \
            self._memory.sample(self.config.batch_size)
        target = self._compute_target(reward, successor)
        if self._sync_target(self.timestep):
            self._target.weights = self._actor.weights
        self._actor.set_option('learning_rate',
            self._learning_rate(self.timestep))
        cost = self._actor.train('cost',
            state=state, action=action, target=target)
        self._costs.append(cost)

    def _create_network(self, model):
        # Percetion.
        state = model.add_input('state', self.states.shape)
        hidden = getattr(networks, self.config.network)(model, state)
        values = dense(hidden, self.actions.n, tf.identity)
        values = model.add_output('values', values)
        # Outputs.
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, self.actions.n)
        target = model.add_input('target')
        model.add_output('value', tf.reduce_max(values, 1))
        model.add_output('choice', tf.argmax(values, 1))
        # Training.
        learning_rate = model.add_option(
            'learning_rate', float(self.config.initial_learning_rate))
        model.set_optimizer(self.config.optimizer(
            learning_rate, self.config.rms_decay))
        model.add_cost('cost',
            (tf.reduce_sum(action * values, 1) - target) ** 2)

    def _step(self, state):
        choice, value = self._actor.compute(('choice', 'value'), state=state)
        if not self.training:
            self._maxqs.append(value)
        return choice

    def _compute_target(self, reward, successor):
        terminal = np.isnan(successor.reshape((len(successor), -1))).any(1)
        successor = np.nan_to_num(successor)
        future = self._target.compute('value', state=successor)
        future[terminal] = 0
        target = reward + self.config.discount * future
        return target

    def _log_memory_size(self):
        size = self._memory.nbytes / (1024 ** 3)
        print('Replay memory size', round(size, 2), 'GB')

    @staticmethod
    def _add_preprocesses(trainer, config):
        trainer.add_preprocess(NormalizeReward)
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, config.downsample)
        if config.delta:
            trainer.add_preprocess(Delta)
        trainer.add_preprocess(FrameSkip, config.frame_skip)
        trainer.add_preprocess(NormalizeImage)
