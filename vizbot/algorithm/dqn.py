import numpy as np
import tensorflow as tf
from vizbot.core import Algorithm, Sequential, Policy
from vizbot import model as networks
from vizbot.model import Model, dense
from vizbot.step import (
    Grayscale, Subsample, Maximum, Skip, History, Normalize, NormalizeReward,
    Delta, EpsilonGreedy)
from vizbot.utility import Experience, Decay, Every, merge_dicts


class DQN(Algorithm, Policy):

    """
    Algorithm: Deep Q-Network (DQN)
    Paper: Human-level control through deep reinforcement learning
    Authors: Mnih et al. 2015
    PDF: https://goo.gl/Y3e373

    TODO:
    - Idle for no_op_max=30 timesteps at the beginning of each episode.
    - Use RMSProp parameters from the paper.
    """

    @classmethod
    def defaults(cls):
        # Preprocesses.
        subsample = 2
        frame_skip = 4
        history = 4
        delta = False
        # Exploration.
        epsilon_from = 1.0
        epsilon_to = 0.1
        epsilon_duration = 1e6
        test_epsilon = 0.05
        # Learning.
        network = 'network_dqn_2015'
        replay_capacity = 2e5  # 1e6
        epsilon_after = 5e4
        start_learning = 5e4
        batch_size = 32
        initial_learning_rate = 2.5e-4
        optimizer = tf.train.RMSPropOptimizer
        rms_decay = 0.99
        rms_epsilon = 0.1
        sync_target = 10000
        return merge_dicts(super().defaults(), locals())

    def __init__(self, task, config):
        Algorithm.__init__(self, task, config)
        self._preprocess = self._create_preprocess()
        Policy.__init__(self, self._preprocess.interface)
        self.config.start_learning = int(float(self.config.start_learning))
        # Network.
        self._model = Model(self._create_network)
        self._target = Model(self._create_network)
        self._target.weights = self._model.weights
        self._sync_target = Every(config.sync_target, config.start_learning)
        # print(str(self._model))
        # Learning.
        self._memory = Experience(int(float(config.replay_capacity)))
        self._learning_rate = Decay(
            float(config.initial_learning_rate), 0, self.task.timesteps)
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
        if self.task.directory:
            self._model.save(self.task.directory, 'model')

    def step(self, observation):
        action, value = self._model.compute(
            ('action', 'value'), state=observation)
        if not self.training:
            self._maxqs.append(value)
        return action

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
            self._target.weights = self._model.weights
        self._model.set_option(
            'learning_rate', self._learning_rate(self.timestep))
        cost = self._model.train(
            'cost', state=state, action=action, target=target)
        self._costs.append(cost)

    @property
    def policy(self):
        policy = Sequential(self.task.interface)
        policy.add(self._preprocess)
        policy.add(self)
        return policy

    def _create_preprocess(self):
        policy = Sequential(self.task.interface)
        if self.config.history:
            policy.add(Grayscale)
        subsample = self.config.subsample
        policy.add(Subsample, (subsample, subsample, 1))
        policy.add(Maximum, 2)
        if self.config.frame_skip:
            policy.add(Skip, self.config.frame_skip)
        if self.config.history:
            policy.add(History, self.config.history)
        policy.add(NormalizeReward)
        if self.config.delta:
            policy.add(Delta)
        policy.add(Normalize)
        policy.add(NormalizeReward)
        policy.add(
            EpsilonGreedy, self.config.epsilon_from, self.config.epsilon_to,
            self.config.epsilon_after, self.config.epsilon_to)
        return policy

    def _create_network(self, model):
        observations, actions = self._preprocess.interface
        # Percetion.
        state = model.add_input('state', self.observations.shape)
        hidden = getattr(networks, self.config.network)(model, state)
        values = dense(hidden, actions.n, tf.identity)
        values = model.add_output('values', values)
        # Outputs.
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, self.actions.n)
        target = model.add_input('target')
        model.add_output('value', tf.reduce_max(values, 1))
        model.add_output('action', tf.argmax(values, 1))
        # Training.
        learning_rate = model.add_option(
            'learning_rate', float(self.config.initial_learning_rate))
        model.set_optimizer(self.config.optimizer(
            learning_rate=learning_rate,
            decay=self.config.rms_decay,
            epsilon=self.config.rms_epsilon))
        model.add_cost(
            'cost', (tf.reduce_sum(action * values, 1) - target) ** 2)

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
