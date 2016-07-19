import time
from threading import Thread
import numpy as np
import tensorflow as tf
from vizbot.core import Agent, Model
from vizbot.agent import EpsilonGreedy
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, Experience, dense, conv2d, rnn


class Async(Agent):

    @staticmethod
    def _config():
        discount = 0.99
        downsample = 4
        frame_skip = 4
        sync_target = 40000
        heads = 32
        apply_gradient = 5
        epsilon = [
            AttrDict(start=1, stop=0.10, over=4e6),
            AttrDict(start=1, stop=0.01, over=4e6),
            AttrDict(start=1, stop=0.50, over=4e6)]
        optimizer = (tf.train.RMSPropOptimizer, 1e-4, 0.99)
        return AttrDict(**locals())

    def __init__(self, trainer):
        self._config = self._config()
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, self._config.downsample)
        trainer.add_preprocess(FrameSkip, self._config.frame_skip)
        super().__init__(trainer)
        self.actor = Model(self._create_network, self._config.optimizer)
        self.target = Model(self._create_network)
        self.target.weights = self.actor.weights
        self._threads = self._create_threads()

    def __call__(self):
        for thread in self._threads:
            thread.start()
        while self._trainer.running:
            if self._trainer.timestep % self._config.sync_target == 0:
                self.target.weights = self.actor.weights
                print('Update target network')
            time.sleep(0.01)
        for thread in self._threads:
            thread.join()

    def _create_threads(self):
        for index in range(self._config.heads):
            agent = Head(self._trainer, self, self._config)
            thread = Thread(None, agent, 'head-{}'.format(index))
            yield thread

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
        print('Average cost', sum(self._costs) / len(self._costs))
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
