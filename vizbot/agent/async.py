import os
import time
from threading import Thread
import numpy as np
import tensorflow as tf
from vizbot.core import Agent
from vizbot.agent import EpsilonGreedy
from vizbot.model import Model, DivergedError, dense, default_network
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, Experience, Every, merge_dicts


class Async(Agent):

    @classmethod
    def _config(cls):
        discount = 0.99
        downsample = 2
        frame_skip = 4
        heads = 32
        sync_target = int(4e4)
        apply_gradient = 5
        epsilon = [
            AttrDict(start=1, stop=0.10, over=4e6),
            AttrDict(start=1, stop=0.01, over=4e6),
            AttrDict(start=1, stop=0.50, over=4e6)]
        optimizer = (tf.train.RMSPropOptimizer, 1e-4)
        save_model = int(1e5)
        load_dir = ''
        return locals()

    def __init__(self, trainer, **config):
        self._config = AttrDict(merge_dicts(self._config(), config))
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, self._config.downsample)
        trainer.add_preprocess(FrameSkip, self._config.frame_skip)
        super().__init__(trainer)
        self.actor = Model(self._create_network, self._config.optimizer,
                self._config.load_dir)
        self.target = Model(self._create_network, None, self._config.load_dir)
        self.target.weights = self.actor.weights
        self._threads = self._create_threads()
        print(str(self.actor))

    def __call__(self):
        for thread in self._threads:
            thread.start()
        sync_target = Every(self._config.sync_target)
        save_model = Every(self._config.save_model)
        while self._trainer.running:
            if sync_target(self._trainer.timestep):
                self.target.weights = self.actor.weights
            if save_model(self._trainer.timestep) and self._trainer.directory:
                self.actor.save(os.path.join(self._trainer.directory, 'actor'))
            time.sleep(0.01)
        for thread in self._threads:
            thread.join()

    def _create_threads(self):
        threads = []
        for index in range(self._config.heads):
            agent = Head(self._trainer, self, **self._config)
            thread = Thread(None, agent, 'head-{}'.format(index))
            threads.append(thread)
        return threads

    def _create_network(self, model):
        raise NotImplementedError


class Head(EpsilonGreedy):

    def __init__(self, trainer, master, **config):
        self._config = AttrDict(config)
        epsilon = master._random.choice(self._config.epsilon)
        super().__init__(trainer, **epsilon)
        self._master = master
        self._batch = Experience(self._config.apply_gradient)
        self._costs = []

    def _step(self, state):
        return self._master.actor.compute('choice', state=state)

    def stop(self):
        if len(self._costs) < 2500:
            return
        print('Cost {:8.2f}'.format(sum(self._costs) / len(self._costs)))
        self._costs = []

    def experience(self, state, action, reward, successor):
        self._batch.append((state, action, reward, successor))
        done = (successor is None)
        if not done and len(self._batch) < self._config.apply_gradient:
            return
        state, action, reward, successor = self._batch.access()
        self._batch.clear()
        target = self._compute_target(reward, successor)
        try:
            cost = self._master.actor.train(
                'cost', state=state, action=action, target=target)
        except DivergedError:
            print('Cost diverged')
        self._costs.append(cost)

    def _compute_target(self, reward, successor):
        future = self._master.target.compute('value', state=successor)
        final = np.isnan(successor.reshape((len(successor), -1))).any(1)
        future[final] = 0
        target = reward + self._config.discount * future
        return target

    def _collect_delta(self, gradient):
        if not self._delta:
            self._delta = gradient
            return
        assert self._delta.keys() == gradient.keys()
        self._delta = {k: self._delta[k] + gradient[k] for k in gradient}


class Q(Async):

    @classmethod
    def _config(cls):
        load_dir = ''
        return merge_dicts(super()._config(), locals())

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



class SARSA(Async):

    @classmethod
    def _config(cls):
        load_dir = ''
        return merge_dicts(super()._config(), locals())

    def _create_network(self, model):
        state = model.add_input('state', self.states.shape)
        action = model.add_input('action', self.actions.shape)
        target = model.add_input('target')
        values = dense(default_network(state), self.actions.shape, tf.identity)
        policy = tf.nn.softmax(values)
        model.add_output('value', tf.reduce_sum(values * policy, 1))
        sample = tf.squeeze(tf.multinomial(policy, 1), (1,))
        model.add_output('choice', tf.one_hot(sample, self.actions.shape))
        model.add_cost('cost',
            (tf.reduce_sum(action * values, 1) - target) ** 2)
