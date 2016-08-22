import numpy as np
import tensorflow as tf
from vizbot.core import Algorithm, Policy, Sequential
from vizbot import model as networks
from vizbot.model import Model, dense
from vizbot.step import (
    Grayscale, Subsample, Maximum, Skip, History, Normalize, ClampReward,
    Delta, Experience)
from vizbot.utility import AttrDict, Experience as Memory, Decay, merge_dicts


class A3C(Algorithm):

    """
    Algorithm: Asynchronous Advantage Actor Critic (A3C)
    Paper: Asynchronous Methods for Deep Reinforcement Learning
    Authors: Mnih et al. 2016
    PDF: https://arxiv.org/pdf/1602.01783v2.pdf
    """

    @classmethod
    def defaults(cls):
        # Preprocessing.
        subsample = 2
        frame_skip = 4
        history = 4
        delta = False
        frame_max = 2
        # Architecture.
        learners = 16
        apply_gradient = 5
        network = 'network_a3c_lstm'
        scale_critic_loss = 0.5
        regularize = 0.01
        # Optimizer.
        initial_learning_rate = 7e-4
        optimizer = tf.train.RMSPropOptimizer
        rms_decay = 0.99
        rms_epsilon = 0.1
        return merge_dicts(super().defaults(), locals())

    def __init__(self, task, config):
        super().__init__(task, config)
        self._preprocess = self._create_preprocess()
        self.model = Model(self._create_network)
        # print(str(self.model))
        self.learning_rate = Decay(
            float(config.initial_learning_rate), 0, self.task.steps)
        self.costs = None
        self.values = None
        self.choices = None

    @property
    def train_policies(self):
        trainers = []
        for _ in range(self.config.learners):
            config = AttrDict(self.config.copy())
            model = Model(self._create_network, threads=1)
            model.weights = self.model.weights
            policy = Sequential(self.task)
            policy.add(self._create_preprocess())
            policy.add(Train, config, self, model)
            trainers.append(policy)
        return trainers

    @property
    def test_policy(self):
        policy = Sequential(self.task)
        policy.add(self._preprocess)
        policy.add(Test, self.model)
        return policy

    def begin_epoch(self):
        super().begin_epoch()
        self.costs = []
        self.values = []
        self.choices = []

    def end_epoch(self):
        super().end_epoch()
        if self.costs:
            average = sum(self.costs) / len(self.costs)
            print('Cost  {:12.5f}'.format(average))
        if self.values:
            average = sum(self.values) / len(self.values)
            print('Value {:12.5f}'.format(average))
        if self.choices:
            dist = np.bincount(self.choices) / len(self.choices)
            dist = ' '.join('{:.2f}'.format(x) for x in dist)
            print('Choices [{}]'.format(dist))
        if self.task.directory:
            self.model.save(self.task.directory, 'model')

    def _create_network(self, model):
        observs = self._preprocess.above_task.observs
        actions = self._preprocess.above_task.actions
        # Perception.
        state = model.add_input('state', observs.shape)
        hidden = getattr(networks, self.config.network)(model, state)
        value = model.add_output(
            'value', tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(value, actions.n, tf.nn.softmax)
        model.add_output(
            'choice', tf.squeeze(tf.multinomial(tf.log(policy), 1), [1]))
        # Objectives.
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions.n)
        return_ = model.add_input('return_')
        logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
        entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
        advantage = tf.stop_gradient(return_ - value)
        actor = advantage * logprob + self.config.regularize * entropy
        critic = self.config.scale_critic_loss * (return_ - value) ** 2 / 2
        # Training.
        learning_rate = model.add_option(
            'learning_rate', float(self.config.initial_learning_rate))
        model.set_optimizer(self.config.optimizer(
            learning_rate, self.config.rms_decay))
        model.add_cost('cost', critic - actor)

    def _create_preprocess(self):
        policy = Sequential(self.task)
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
        policy.add(Normalize)
        return policy


class Train(Experience):

    def __init__(self, task, config, master, model):
        super().__init__(task)
        self._config = config
        self._master = master
        self._model = model
        self._batch = Memory(self._config.apply_gradient)
        self._context_last_batch = None

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._context_last_batch = None
        if self._model.has_option('context'):
            self._model.reset_option('context')

    def perform(self, observ):
        assert self.training
        choice, value = self._model.compute(
            ('choice', 'value'), state=observ)
        self._master.choices.append(choice)
        self._master.values.append(value)
        return choice

    def experience(self, observ, action, reward, successor):
        self._batch.append((observ, action, reward, successor))
        done = (successor is None)
        if not done and len(self._batch) < self._config.apply_gradient:
            return
        return_ = (
            0 if done else self._model.compute('value', state=observ))
        self._train(self._batch.access(), return_)
        self._batch.clear()
        self._model.weights = self._master.model.weights

    def _train(self, transitions, return_):
        observs, actions, rewards, _ = transitions
        returns = self._compute_eligibilities(rewards, return_)
        self._decay_learning_rate()
        if self._model.has_option('context'):
            context = self._context_last_batch
            if context is not None:
                self._model.set_option('context', context)
            else:
                self._model.reset_option('context')
        delta, cost = self._model.delta(
            'cost', action=actions, state=observs, return_=returns)
        # self._log_gradient(delta)
        self._master.model.apply(delta)
        self._master.costs.append(cost)
        if self._model.has_option('context'):
            self._context_last_batch = self._model.get_option('context')

    def _compute_eligibilities(self, rewards, return_):
        returns = []
        for reward in reversed(rewards):
            return_ = reward + self._config.discount * return_
            returns.append(return_)
        returns = np.array(list(reversed(returns)))
        return returns

    def _decay_learning_rate(self):
        learning_rate = self._master.learning_rate(self._master.task.step)
        self._model.set_option('learning_rate', learning_rate)  # Remove.
        self._master.model.set_option('learning_rate', learning_rate)

    def _log_gradient(self, delta):
        keys = sorted(delta.keys())
        vals = [delta[x].mean() for x in keys]
        for key, val in zip(keys, vals):
            print('{:<40} {:20.16f}'.format(key, val))


class Test(Policy):

    def __init__(self, task, model):
        super().__init__(task)
        self._model = model

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        if self._model.has_option('context'):
            self._model.reset_option('context')

    def observe(self, observ):
        super().observe(observ)
        assert not self.training
        return self._model.compute('choice', state=observ)

    def receive(self, reward, final):
        super().receive(reward, final)
