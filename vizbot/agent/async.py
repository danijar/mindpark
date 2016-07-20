import os
import time
from threading import Thread
import numpy as np
import tensorflow as tf
from vizbot.core import Agent
from vizbot.agent import EpsilonGreedy
from vizbot.model import Model, dense, conv2d
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, Experience


class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, current):
        if self._last is None:
            self._last = current
        if current - self._last < self._every:
            return False
        self._last += self._every
        return True


class Async(Agent):

    @staticmethod
    def _config():
        discount = 0.99
        downsample = 4
        frame_skip = 4
        # sync_target = 40000
        sync_target = 2000
        heads = 16
        apply_gradient = 5
        epsilon = [
            AttrDict(start=1, stop=0.10, over=4e6),
            AttrDict(start=1, stop=0.01, over=4e6),
            AttrDict(start=1, stop=0.50, over=4e6)]
        optimizer = (tf.train.RMSPropOptimizer, 5e-4, 0.99)
        # save_model = int(1e5)
        save_model = 2000
        load_dir = \
            '~/experiment/gym/2016-07-20T00-56-57-experiment/' \
            'SimpleDoom-v0-Async/repeat-0/actor'
        return AttrDict(**locals())

    def __init__(self, trainer):
        self._config = self._config()
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
        print('Launch threads')
        for thread in self._threads:
            thread.start()
        print('Start training')
        sync_target = Every(self._config.sync_target)
        save_model = Every(self._config.save_model)
        while self._trainer.running:
            if sync_target(self._trainer.timestep):
                print('Update target network')
                self.target.weights = self.actor.weights
            if save_model(self._trainer.timestep) and self._trainer.directory:
                print('Store model')
                self.actor.save(os.path.join(self._trainer.directory, 'actor'))
            time.sleep(0.01)
        print('Close threads')
        for thread in self._threads:
            thread.join()

    def _create_threads(self):
        threads = []
        for index in range(self._config.heads):
            agent = Head(self._trainer, self, self._config)
            thread = Thread(None, agent, 'head-{}'.format(index))
            threads.append(thread)
        return threads

    def _create_network(self, model):
        state = model.add_input('state', self.states.shape)
        action = model.add_input('action', self.actions.shape)
        target = model.add_input('target')
        x = conv2d(state, 16, 4, 3, pool=2)
        x = conv2d(x, 32, 2, 1)
        x = dense(x, 256)
        x = dense(x, 256)
        x = dense(x, self.actions.shape, tf.identity)
        model.add_output('best', tf.reduce_max(x, 1))
        model.add_output('act', tf.one_hot(tf.argmax(x,1), self.actions.shape))
        model.add_cost('cost', (tf.reduce_sum(action * x, 1) - target) ** 2)


class Head(EpsilonGreedy):

    def __init__(self, trainer, master, config):
        self._config = config
        epsilon = master._random.choice(config.epsilon)
        super().__init__(trainer, **epsilon)
        self._master = master
        self._timestep = 0
        self._batch = Experience(self._config.apply_gradient)
        self._costs = []

    def _step(self, state):
        return self._master.actor.compute('act', state=state)

    def stop(self):
        if len(self._costs) < 100:
            return
        print('Cost', sum(self._costs) / len(self._costs))
        self._costs = []

    def experience(self, state, action, reward, successor):
        self._timestep += 1
        self._batch.append((state, action, reward, successor))
        done = (successor is None)
        if not done and len(self._batch) < self._config.apply_gradient:
            return
        state, action, reward, successor = self._batch.access()
        self._batch.clear()
        target = self._compute_target(reward, successor)
        cost = self._master.actor.train(
            'cost', state=state, action=action, target=target)
        self._costs.append(cost)

    def _compute_target(self, reward, successor):
        future = self._master.target.compute('best', state=successor)
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
