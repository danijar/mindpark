from threading import Thread, Lock
import numpy as np
import tensorflow as tf
from vizbot.core import Agent
from vizbot.agent import EpsilonGreedy
from vizbot.utility import AttrDict


class Asnyc(Agent):

    @staticmethod
    def _config():
        discount = 0.99
        sync_target = 40000
        heads = 16
        apply_gradient = 5
        epsilon = [
            AttrDict(start=1, stop=0.10, over=4e6),
            AttrDict(start=1, stop=0.01, over=4e6),
            AttrDict(start=1, stop=0.50, over=4e6)]
        learning_rate = 1e-4
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99)
        return AttrDict(**locals())

    def __init__(self, trainer):
        super().__init__(trainer)
        self._config = self._config()
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, self._config.downsample)
        trainer.add_preprocess(FrameSkip, self._config.frame_skip)
        with Model() as model:
            self.actor = self._create_network(model)
        with Model() as model:
            self.target = self._create_network(model)
        self._threads = self._create_threads()
        self._target.variables = self._actor.variables
        self._lock = Lock()

    def __call__(self):
        for thread in self._threads:
            thread.start()
        while self._trainer.running:
            if self._trainer.timestep % self._config.sync_target == 0:
                self._target.variables = self._actor.variables
            time.sleep(0.01)
        for thread in self._threads:
            thread.join()

    def apply_gradient(self, gradient):
        with self._lock:
            raise NotImplementedError

    def _create_threads(self):
        for index in range(self._config.heads):
            epsilon = self._random.choice(self._config.epsilon)
            agent = Head(self._trainer, self, self._config)
            thread = Thread(None, agent, 'head-{}'.format(index))
            yield thread

    def _create_network(self, model):
        model.placeholder('state', self.states.shape)
        model.placeholder('action_', self.actions.shape)
        model.placeholder('target')
        activation = tf.nn.elu
        x = conv2d(model.state, 16, 4, 3, activation, 2)
        x = conv2d(x, 32, 2, 1, activation)
        x = dense(x, 256, activation)
        x = dense(x, 256, activation)
        x = dense(x, self.actions.shape, activation)
        cost = (tf.reduce_sum(model.action_ * x, 1) - model.target) ** 2
        gradient = self._config.optimizer.compute_gradients(cost)
        gradient, variables = zip(*gradient)
        model.action('best', tf.reduce_max(x, 1))
        model.action('act', tf.one_hot(tf.argmax(x, 1), self.actions.shape))
        model.action('gradient', gradient)
        model.action('apply', gradient)
        # TODO: Add losses independently from compiling. Rename compile() to
        # initizlize().
        model.compile(cost, self._config.optimizer)
        return model, variables


class Head(EpsilonGreedy):

    def __init__(self, trainer, master, config):
        self._config = config
        super().__init__(trainer, **config._epsilon)
        self._master = master
        self._timestep = 0
        self._gradient = None

    def _step(self, state):
        return self._master.actor.perform(state=state)

    def experience(self, state, action, reward, successor):
        self._timestep += 1
        target = self._compute_target(reward, successor)
        gradient = self._master.actor.gradient(
            state=state, action=action, target=target)
        self._add_gradient(gradient)
        terminal = (successor is None)
        if self._timestep % self._config.apply_gradient == 0 or terminal:
            self._master.apply_gradient(self._gradient)
            self._gradient = None

    def _compute_target(reward, successor):
        future = self._master.target.best(state=successor)
        final = np.isnan(successor.reshape((len(successor), -1))).any(1)
        future[final] = 0
        target = reward + self._config.discount * future
        return target

    def _add_gradient(self, gradient):
        if self._gradient is None:
            self._gradient = gradient
            return
        # TODO: Not as easy as this...
        self._gradient += gradient
