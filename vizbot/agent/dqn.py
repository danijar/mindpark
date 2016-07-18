import collections
import numpy as np
import tensorflow as tf
from vizbot.core import Model
from vizbot.agent import EpsilonGreedy
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, Experience, lazy_property, dense, conv2d


class DQN(EpsilonGreedy):

    def __init__(self, trainer, config=None):
        self._config = config or self._default_config()
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, self._config.downsample)
        trainer.add_preprocess(FrameSkip, self._config.frame_skip)
        super().__init__(trainer, **self._config.epsilon)
        self._memory = Experience(self._config.replay_capacity)
        with Model() as model:
            self._actor = self._build_q_network(model)
        with Model() as model:
            self._target = self._build_q_network(model)
        self._target.variables = self._actor.variables

    def _step(self, state):
        return self._actor.perform(state=state)

    def experience(self, state, action, reward, successor):
        self._memory.append(state, action, reward, successor)
        print('Replay memory size', self._memory.nbytes / 1024 / 1024, 'MB')
        if len(self._memory) < self._config.batch_size:
            return
        state, action, reward, successor = \
            self._memory.sample(self._config.batch_size)
        future = self._target.best(state=successor)
        final = np.isnan(successor.reshape((len(successor), -1))).any(1)
        future[final] = 0
        target = reward + self._config.discount * future
        self._target.variables = self._actor.variables
        self._actor.train(state=state, action_=action, target=target)

    def _build_q_network(self, model):
        model.placeholder('state', self._env.states.shape)
        model.placeholder('action_', self._env.actions.shape)
        model.placeholder('target')
        x = conv2d(model.state, 16, 8, 4, tf.nn.relu)
        x = conv2d(x, 32, 4, 3, tf.nn.relu)
        x = dense(x, 256, tf.nn.relu)
        x = dense(x, self._env.actions.shape, tf.nn.relu)
        cost = (tf.reduce_sum(model.action_ * x, 1) - model.target) ** 2
        model.action('best', tf.reduce_max(x, 1))
        model.action('perform',
            tf.one_hot(tf.argmax(x, 1), self._env.actions.shape))
        model.compile(cost, self._config.optimizer)
        return model

    @staticmethod
    def _default_config():
        discount = 0.99
        downsample = 4
        frame_skip = 4
        replay_capacity = int(1e5)
        batch_size = 32
        learning_rate = 1e-4
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # epsilon = AttrDict(start=1, stop=0.1, over=int(1e6))
        epsilon = AttrDict(start=0.8, stop=0, over=int(1e5))
        return AttrDict(**locals())
