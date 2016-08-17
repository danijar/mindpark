import numpy as np
import tensorflow as tf
from vizbot.core import Algorithm, Sequential
from vizbot import model as networks
from vizbot.model import Model, dense
from vizbot.step import (
    Grayscale, Subsample, Maximum, Skip, History, Normalize, ClampReward,
    Delta, EpsilonGreedy, Experience)
from vizbot.utility import Experience as Memory, Decay, Every, merge_dicts


class DQN(Algorithm, Experience):

    """
    Algorithm: Deep Q-Network (DQN)
    Paper: Human-level control through deep reinforcement learning
    Authors: Mnih et al. 2015
    PDF: https://goo.gl/Y3e373
    """

    # TODO: Skip up to 30 env steps. Not sure if only during evalution or also
    # during training. Both seems like a good idea.

    @classmethod
    def defaults(cls):
        # Preprocessing.
        subsample = 2
        frame_skip = 4
        history = 4
        delta = False
        frame_max = 2
        # Architecture.
        network = 'network_dqn_2015'
        replay_capacity = 1e5  # 1e6
        start_learning = 5e4
        # Exploration.
        epsilon = dict(
            from_=1.0, to=0.1, test=0.05, over=1e6, offset=start_learning)
        # Learning.
        batch_size = 32
        sync_target = 2500
        # Optimizer.
        initial_learning_rate = 2.5e-4
        optimizer = tf.train.RMSPropOptimizer
        rms_decay = 0.95
        rms_epsilon = 0.1
        return merge_dicts(super().defaults(), locals())

    def __init__(self, task, config):
        Algorithm.__init__(self, task, config)
        self._preprocess = self._create_preprocess()
        Experience.__init__(self, self._preprocess.interface)
        # Parse parameters (until YAML 1.2 support).
        self.config.start_learning = int(float(self.config.start_learning))
        self.config.epsilon = {
            k: float(v) for k, v in self.config.epsilon.items()}
        # Network.
        self._model = Model(self._create_network)
        self._target = Model(self._create_network)
        self._target.weights = self._model.weights
        self._sync_target = Every(config.sync_target, config.start_learning)
        # print(str(self._model))
        # Learning.
        self._memory = Memory(int(float(config.replay_capacity)))
        self._learning_rate = Decay(
            float(config.initial_learning_rate), 0, self.task.steps)
        self._costs = None
        self._maxqs = None

    def begin_epoch(self, epoch):
        super().begin_epoch(epoch)
        self._costs = []
        self._maxqs = []

    def end_epoch(self):
        super().end_epoch()
        if self._costs:
            print('Cost {:8.3f}'.format(sum(self._costs) / len(self._costs)))
        if self._maxqs:
            print('MaxQ {:8.3f}'.format(sum(self._maxqs) / len(self._maxqs)))
        if self.task.directory:
            self._model.save(self.task.directory, 'model')

    def perform(self, observation):
        action, value = self._model.compute(
            ('action', 'value'), state=observation)
        if not self.training:
            self._maxqs.append(value)
        return action

    def experience(self, observation, action, reward, successor):
        self._memory.append((observation, action, reward, successor))
        if len(self._memory) == 1:
            self._log_memory_size()
        if len(self._memory) < self.config.start_learning:
            return
        observation, action, reward, successor = \
            self._memory.sample(self.config.batch_size)
        target = self._compute_target(reward, successor)
        if self._sync_target(self.step):
            self._target.weights = self._model.weights
        self._model.set_option(
            'learning_rate', self._learning_rate(self.task.step))
        cost = self._model.train(
            'cost', state=observation, action=action, target=target)
        self._costs.append(cost)

    @property
    def policy(self):
        policy = Sequential(self.task.interface)
        policy.add(self._preprocess)
        policy.add(self)
        return policy

    def _create_preprocess(self):
        policy = Sequential(self.task.interface)
        if self.config.frame_skip:
            policy.add(Skip, self.config.frame_skip)
        if self.config.frame_max:
            policy.add(Maximum, self.config.frame_max)
        if self.config.history:
            policy.add(Grayscale)
        if self.config.subsample > 1:
            sub = self.config.subsample
            amount = (sub, sub) if self.config.history else (sub, sub, 1)
            policy.add(Subsample, amount)
        if self.config.delta:
            policy.add(Delta)
        if self.config.history:
            policy.add(History, self.config.history)
        policy.add(ClampReward)
        policy.add(EpsilonGreedy, **self.config.epsilon)
        policy.add(Normalize)
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
        assert np.isfinite(successor).all()
        future = self._target.compute('value', state=successor)
        future[terminal] = 0
        target = reward + self.config.discount * future
        assert np.isfinite(target).all()
        return target

    def _log_memory_size(self):
        size = self._memory.nbytes / (1024 ** 3)
        print('Replay memory size', round(size, 2), 'GB')
