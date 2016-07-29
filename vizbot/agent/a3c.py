import os
import time
from threading import Thread, Lock
import numpy as np
import tensorflow as tf
from vizbot.core import Agent
from vizbot import model as networks
from vizbot.model import Model, dense
from vizbot.preprocess import (
    Grayscale, Downsample, FrameSkip, NormalizeReward, NormalizeImage)
from vizbot.utility import (
    AttrDict, Experience, Decay, merge_dicts, lazy_property)


class A3C(Agent):

    @classmethod
    def defaults(cls):
        network = 'network_a3c_lstm'
        # Preprocesses.
        downsample = 2
        frame_skip = 4
        # Learning.
        learners = 16
        apply_gradient = 5
        regularize = 0.01
        initial_learning_rate = 7e-4
        optimizer = tf.train.RMSPropOptimizer
        rms_decay = 0.99
        scale_critic_loss = 0.5
        # Logging.
        return merge_dicts(super().defaults(), locals())

    def __init__(self, trainer, config):
        self._add_preprocesses(trainer, config)
        super().__init__(trainer, config)
        self.model = Model(self._create_network)
        # print(str(self.model))
        self.learning_rate = Decay(
            float(config.initial_learning_rate), 0, self._trainer.timesteps)
        self.costs = []
        self.values = []
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

    def stop_epoch(self):
        super().stop_epoch()
        with self.lock:
            if self.costs:
                average = sum(self.costs) / len(self.costs)
                print('Cost  {:12.5f}'.format(average))
                self.costs = []
            if self.values:
                average = sum(self.values) / len(self.values)
                print('Value {:12.5f}'.format(average))
                self.costs = []
        if self._trainer.directory:
            self.model.save(self._trainer.directory, 'model')

    def _create_network(self, model):
        # Perception.
        state = model.add_input('state', self.states.shape)
        hidden = getattr(networks, self.config.network)(model, state)
        value = model.add_output('value',
            tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(hidden, self.actions.n, tf.nn.softmax)
        sample = tf.squeeze(tf.multinomial(tf.log(policy), 1), [1])
        model.add_output('choice', sample)
        # Objectives.
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, self.actions.n)
        return_ = model.add_input('return_')
        advantage = return_ - value
        logprob = tf.log(tf.reduce_max(policy * action, 1) + 1e-9)
        entropy = -tf.reduce_sum(tf.log(policy + 1e-9) * policy)
        actor = logprob * advantage + self.config.regularize * entropy
        critic = self.config.scale_critic_loss * (return_ - value) ** 2 / 2
        # Training.
        learning_rate = model.add_option(
            'learning_rate', float(self.config.initial_learning_rate))
        model.set_optimizer(self.config.optimizer(
            learning_rate, self.config.rms_decay))
        model.add_cost('cost', tf.reduce_mean(critic) - tf.reduce_mean(actor))

    @staticmethod
    def _add_preprocesses(trainer, config):
        trainer.add_preprocess(NormalizeReward)
        trainer.add_preprocess(NormalizeImage)
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, config.downsample)
        trainer.add_preprocess(FrameSkip, config.frame_skip)


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
        self._master.values.append(value)
        return choice

    def experience(self, state, action, reward, successor):
        self._batch.append((state, action, reward, successor))
        done = (successor is None)
        if not done and len(self._batch) < self.config.apply_gradient:
            return
        return_= 0 if done else self._master.model.compute('value',state=state)
        self._train(self._batch.access(), return_)
        self._batch.clear()
        self._model.weights = self._master.model.weights

    def _train(self, transitions, return_):
        states, actions, rewards, _ = transitions
        returns = self._compute_eligibilities(rewards, return_)
        self._decay_learning_rate()
        if self._master.model.has_option('context'):
            context = self._context_last_batch
            if context is not None:
                self._master.model.set_option('context', context)
            else:
                self._master.model.reset_option('context')
        delta, cost = self._master.model.delta('cost',
            action=actions, state=states, return_=returns)
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
        self._master.model.set_option('learning_rate',
            self._master.learning_rate(self.timestep))


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
