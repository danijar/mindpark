import os
import time
from threading import Thread
import numpy as np
import tensorflow as tf
from vizbot.core import Agent
from vizbot.agent import EpsilonGreedy
from vizbot import model as networks
from vizbot.model import Model, dense
from vizbot.preprocess import (
    Grayscale, Downsample, FrameSkip, NormalizeReward, NormalizeImage)
from vizbot.utility import (
    AttrDict, Experience, Every, Statistic, Decay, merge_dicts)


class A3C(Agent):

    @classmethod
    def defaults(cls):
        network = 'network_dqn'
        # Preprocesses.
        downsample = 2
        frame_skip = 4
        # Exploration.
        heads = 16
        epsilon_from = 1.0
        epsilon_tos = [0.1, 0.01, 0.5]
        epsilon_duration = 5e5
        # Learning.
        apply_gradient = 5
        regularize = 0.01
        initial_learning_rate = 1e-4
        optimizer = tf.train.RMSPropOptimizer
        rms_decay= 0.99
        scale_critic_loss = 0.5
        # Logging.
        save_model = int(1e5)
        print_cost = int(5e4)
        load_dir = ''
        return merge_dicts(super().defaults(), locals())

    def __init__(self, trainer, config):
        self._add_preprocesses(trainer, config)
        super().__init__(trainer, config)
        self.model = Model(self._create_network)
        print(str(self.model))
        self.costs = Statistic('Cost {:12.5f}', config.print_cost)
        self.learning_rate = Decay(
            float(config.initial_learning_rate), 0, self._trainer._timesteps)
        self._threads = self._create_threads()

    def __call__(self):
        for thread in self._threads:
            thread.start()
        save_model = Every(self.config.save_model)
        while self._trainer.running:
            if save_model(self._trainer.timestep) and self._trainer.directory:
                self.model.save(self._trainer.directory, 'model')
            time.sleep(0.01)
        for thread in self._threads:
            thread.join()

    def _add_preprocesses(self, trainer, config):
        trainer.add_preprocess(NormalizeReward)
        trainer.add_preprocess(NormalizeImage)
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, config.downsample)
        trainer.add_preprocess(FrameSkip, config.frame_skip)

    def _create_network(self, model):
        # Perception.
        state = model.add_input('state', self.states.shape)
        hidden = getattr(networks, self.config.network)(model, state)
        value = model.add_output('value',
            tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(hidden, self.actions.shape, tf.nn.softmax)
        sample = tf.squeeze(tf.multinomial(policy, 1), [1])
        model.add_output('choice', tf.one_hot(sample, self.actions.shape))
        # Objectives.
        action = model.add_input('action', self.actions.shape)
        return_ = model.add_input('return_')
        advantage = return_ - value
        gradient = tf.reduce_sum(tf.log(policy * action + 1e-9))
        entropy = tf.reduce_sum(tf.log(policy + 1e-9) * policy)
        actor = gradient * advantage + self.config.regularize * entropy
        critic = self.config.scale_critic_loss * (return_ - value) ** 2 / 2
        # Training.
        learning_rate = model.add_option(
            'learning_rate', float(self.config.initial_learning_rate))
        model.set_optimizer(self.config.optimizer(
            learning_rate, self.config.rms_decay))
        model.add_cost('cost', tf.reduce_sum(critic) - tf.reduce_sum(actor))

    def _create_threads(self):
        threads = []
        for index in range(self.config.heads):
            config = self.config.copy()
            config['epsilon_to'] = self._random.choice(self.config.epsilon_tos)
            model = Model(self._create_network)
            model.weights = self.model.weights
            agent = Head(self._trainer, self, model, AttrDict(config))
            thread = Thread(None, agent, 'head-{}'.format(index))
            threads.append(thread)
        return threads


class Head(EpsilonGreedy):

    def __init__(self, trainer, master, model, config):
        super().__init__(trainer, config)
        self._master = master
        self._model = model
        self._batch = Experience(self.config.apply_gradient)
        self._context_last_batch = None
        self._reset_context()

    def __call__(self):
        while self._trainer.running:
            self._reset_context()
            self._trainer.run_episode(self, self._env)

    def _step(self, state):
        return self._model.compute('choice', state=state)

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
        current = self._stash_context()
        self._decay_learning_rate()
        self._master.costs(self._master.model.train('cost',
            action=actions, state=states, return_=returns))
        self._unstash_context(current)
        self._context_last_batch = current

    def _compute_eligibilities(self, rewards, return_):
        returns = []
        for reward in reversed(rewards):
            return_ = reward + self.config.discount * return_
            returns.append(return_)
        returns = np.array(list(reversed(returns)))
        return returns

    def _reset_context(self):
        if not self._model.has_option('context'):
            return
        self._model.reset_option('context')
        self._context_last_batch = self._model.get_option('context')

    def _stash_context(self):
        if not self._model.has_option('context'):
            return
        current = self._model.get_option('context')
        self._model.set_option('context', self._context_last_batch)
        return current

    def _unstash_context(self, current):
        if not self._model.has_option('context'):
            return
        self._model.set_option('context', current)
        self._context_last_batch = current

    def _decay_learning_rate(self):
        self._master.model.set_option('learning_rate',
            self._master.learning_rate(self.timestep))
