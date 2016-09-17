import numpy as np
import tensorflow as tf
from mindpark.core import Algorithm, Sequential, Metric
from mindpark import model as networks
from mindpark.model import Model, dense
from mindpark.step import (
    RandomStart, Grayscale, Subsample, Maximum, Skip, History, Normalize,
    ClampReward, Delta, EpsilonGreedy, Experience, Image)
from mindpark.utility import Experience as Memory, Decay, Every, merge_dicts


class DQN(Algorithm, Experience):

    """
    Algorithm: Deep Q-Network (DQN)
    Paper: Human-level control through deep reinforcement learning
    Authors: Mnih et al. 2015
    PDF: https://goo.gl/Y3e373
    """

    @classmethod
    def defaults(cls):
        # Preprocessing.
        subsample = 2
        frame_skip = 4
        history = 4
        delta = False
        frame_max = 2
        noop_max = 30
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
        # Parse parameters (until YAML 1.2 support).
        self.config.start_learning = int(float(self.config.start_learning))
        self.config.sync_target = int(float(self.config.sync_target))
        self.config.epsilon.over = int(float(self.config.epsilon.over))
        self.config.replay_capacity = int(float(self.config.replay_capacity))
        # Scale parameters.
        assert self.config.start_learning <= self.config.replay_capacity
        assert self.config.start_learning >= self.config.batch_size
        self.config.start_learning *= self.config.frame_skip
        self.config.sync_target *= self.config.frame_skip
        self.config.epsilon.over *= self.config.frame_skip
        # Preprocessing.
        self._preprocess = self._create_preprocess()
        Experience.__init__(self, self._preprocess.above_task)
        # Network.
        self._model = Model(self._create_network)
        self._target = Model(self._create_network)
        self._target.weights = self._model.weights
        self._sync_target = Every(
            self.config.sync_target, self.config.start_learning)
        print(str(self._model))
        # Learning.
        observ_shape = self._preprocess.above_task.observs.shape
        shapes = (observ_shape, tuple(), tuple(), observ_shape)
        self._memory = Memory(self.config.replay_capacity, shapes)
        self._log_memory_size()
        self._learning_rate = Decay(
            float(self.config.initial_learning_rate), 0, self.task.steps)
        self._cost_metric = Metric(self.task, 'dqn/cost', 1)
        self._learning_rate_metric = Metric(self.task, 'dqn/learning_rate', 1)

    def end_epoch(self):
        super().end_epoch()
        if self.task.directory:
            self._model.save(self.task.directory, 'model')

    def perform(self, observ):
        return self._model.compute('values', state=observ)

    def experience(self, observ, action, reward, successor):
        action = action.argmax()
        self._memory.append((observ, action, reward, successor))
        if self.task.step < self.config.start_learning:
            return
        observ, action, reward, successor = \
            self._memory.sample(self.config.batch_size)
        target = self._compute_target(reward, successor)
        if self._sync_target(self.task.step):
            self._target.weights = self._model.weights
        self._model.set_option(
            'learning_rate', self._learning_rate(self.task.step))
        cost = self._model.train(
            'cost', state=observ, action=action, target=target)
        self._learning_rate_metric(self._model.get_option('learning_rate'))
        self._cost_metric(cost)

    @property
    def policy(self):
        # TODO: Why doesn't self.task work here?
        policy = Sequential(self._preprocess.task)
        policy.add(self._preprocess)
        policy.add(self)
        return policy

    def _create_preprocess(self):
        policy = Sequential(self.task)
        policy.add(Image)
        if self.config.noop_max:
            policy.add(RandomStart, self.config.noop_max)
        if self.config.frame_skip > 1:
            policy.add(Skip, self.config.frame_skip)
        if self.config.frame_max:
            policy.add(Maximum, self.config.frame_max)
        if self.config.history > 1:
            channels = policy.above_task.observs.shape[-1]
            policy.add(Grayscale, (0.299, 0.587, 0.114)[:channels])
        if self.config.subsample > 1:
            sub = self.config.subsample
            amount = (sub, sub) if self.config.history > 1 else (sub, sub, 1)
            policy.add(Subsample, amount)
        if self.config.delta:
            policy.add(Delta)
        if self.config.history > 1:
            policy.add(History, self.config.history)
        policy.add(Normalize)
        policy.add(ClampReward)
        policy.add(EpsilonGreedy, **self.config.epsilon)
        return policy

    def _create_network(self, model):
        observs = self._preprocess.above_task.observs.shape
        actions = self._preprocess.above_task.actions.shape[0]
        # Percetion.
        state = model.add_input('state', observs)
        hidden = getattr(networks, self.config.network)(model, state)
        values = dense(hidden, actions, tf.identity)
        values = model.add_output('values', values)
        # Training.
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        target = model.add_input('target')
        model.add_output('value', tf.reduce_max(values, 1))
        # Opimization.
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
