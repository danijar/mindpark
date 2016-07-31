import os
import time
from threading import Thread, Lock
import numpy as np
import tensorflow as tf
from vizbot.core import Agent
from vizbot import model as networks
from vizbot.model import Model, dense
from vizbot.preprocess import (
    Grayscale, Downsample, FrameSkip, NormalizeReward, NormalizeImage, Delta)
from vizbot.utility import (
    AttrDict, Experience, Decay, merge_dicts, lazy_property)


class A3C(Agent):

    @classmethod
    def defaults(cls):
        network = 'network_a3c_lstm'
        # Preprocesses.
        downsample = 2
        frame_skip = 6
        delta = True
        # Learning.
        learners = 16
        apply_gradient = 5
        regularize = 0.01
        initial_learning_rate = 7e-4
        optimizer = tf.train.RMSPropOptimizer
        rms_decay = 0.99
        scale_critic_loss = 0.5
        return merge_dicts(super().defaults(), locals())

    def __init__(self, trainer, config):
        self._add_preprocesses(trainer, config)
        super().__init__(trainer, config)
        self.model = Model(self._create_network)
        # print(str(self.model))
        self.learning_rate = Decay(
            float(config.initial_learning_rate), 0, self._trainer.timesteps)
        self.costs = None
        self.values = None
        self.choices = None
        self.lock = Lock()

    @lazy_property
    def learners(self):
        learners = []
        for _ in range(self.config.learners):
            config = AttrDict(self.config.copy())
            model = Model(self._create_network, threads=1)
            model.weights = self.model.weights
            learner = Learner(self._trainer, self, model, config)
            learners.append(learner)
        return learners

    @lazy_property
    def testee(self):
        return Testee(self._trainer, self.model, AttrDict(self.config.copy()))

    def start_epoch(self):
        super().start_epoch()
        self.costs = []
        self.values = []
        self.choices = []

    def stop_epoch(self):
        super().stop_epoch()
        with self.lock:
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
        if self._trainer.directory:
            self.model.save(self._trainer.directory, 'model')

    def _create_network(self, model):
        # Perception.
        state = model.add_input('state', self.states.shape)
        hidden = getattr(networks, self.config.network)(model, state)
        value = model.add_output('value',
            tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(value, self.actions.n, tf.nn.softmax)
        model.add_output('choice',
            tf.squeeze(tf.multinomial(tf.log(policy), 1), [1]))
        # Objectives.
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, self.actions.n)
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

    @staticmethod
    def _add_preprocesses(trainer, config):
        trainer.add_preprocess(NormalizeReward)
        if config.frame_skip:
            trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, config.downsample)
        if config.delta:
            trainer.add_preprocess(Delta)
        if config.frame_skip:
            trainer.add_preprocess(FrameSkip, config.frame_skip)
        trainer.add_preprocess(NormalizeImage)


class Learner(Agent):

    def __init__(self, trainer, master, model, config):
        super().__init__(trainer, config)
        self._master = master
        self._model = model
        self._batch = Experience(self.config.apply_gradient)
        self._context_last_batch = None

    def start_episode(self, training):
        super().start_episode(training)
        self._context_last_batch = None
        if self._model.has_option('context'):
            self._model.reset_option('context')

    def step(self, state):
        assert self.training
        choice, value = self._model.compute(('choice', 'value'), state=state)
        self._master.choices.append(choice)
        self._master.values.append(value)
        return choice

    def experience(self, state, action, reward, successor):
        self._batch.append((state, action, reward, successor))
        done = (successor is None)
        if not done and len(self._batch) < self.config.apply_gradient:
            return
        return_= 0 if done else self._model.compute('value',state=state)
        self._train(self._batch.access(), return_)
        self._batch.clear()
        self._model.weights = self._master.model.weights

    def _train(self, transitions, return_):
        states, actions, rewards, _ = transitions
        returns = self._compute_eligibilities(rewards, return_)
        self._decay_learning_rate()
        if self._model.has_option('context'):
            context = self._context_last_batch
            if context is not None:
                self._model.set_option('context', context)
            else:
                self._model.reset_option('context')
        delta, cost = self._model.delta('cost',
            action=actions, state=states, return_=returns)
        # self._log_gradient(delta)
        self._master.model.apply(delta)
        self._master.costs.append(cost)
        if self._model.has_option('context'):
            self._context_last_batch = self._model.get_option('context')

    def _compute_eligibilities(self, rewards, return_):
        returns = []
        for reward in reversed(rewards):
            return_ = reward + self.config.discount * return_
            returns.append(return_)
        returns = np.array(list(reversed(returns)))
        return returns

    def _decay_learning_rate(self):
        learning_rate = self._master.learning_rate(self.timestep)
        self._model.set_option('learning_rate', learning_rate)  # Remove.
        self._master.model.set_option('learning_rate', learning_rate)

    def _log_gradient(self, delta):
        keys = sorted(delta.keys())
        vals = [delta[x].mean() for x in keys]
        for key, val in zip(keys, vals):
            print('{:<40} {:20.16f}'.format(key, val))


class Testee(Agent):

    def __init__(self, trainer, model, config):
        super().__init__(trainer, config)
        self._model = model

    def start_episode(self, training):
        super().start_episode(training)
        if self._model.has_option('context'):
            self._model.reset_option('context')

    def step(self, state):
        assert not self.training
        return self._model.compute('choice', state=state)
