import collections
import numpy as np
import tensorflow as tf
from vizbot.core import Model
from vizbot.agent import EpsilonGreedy
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, lazy_property


def conv2d(x, filters, size, stride, activation):
    x = tf.contrib.layers.convolution2d(
        x, filters, size, stride, 'VALID', activation)
    return x


def dense(x, size, activation):
    x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
    x = tf.contrib.layers.fully_connected(x, size, activation)
    return x


class DQN(EpsilonGreedy):

    def __init__(self, env, config=None):
        self._config = config or self._default_config()
        env = Grayscale(env)
        env = Downsample(env, 4)
        env = FrameSkip(env, self._config.frame_skip)
        # env = OneHotActions(env)
        super().__init__(env, **self._config.epsilon)
        self._memory = ReplayMemory(self._config.replay_capacity)
        with tf.Graph().as_default():
            self._actor = self._build_q_network('actor')
        # with tf.Graph().as_default():
            self._target = self._build_q_network('target')
        self._target.variables = self._actor.variables

    def _perform(self, state):
        action = self._noop()
        action[self._actor.choice(state=state)] = 1
        return action

    def feedback(self, action, reward):
        super().feedback(action, reward)
        if reward:
            print('Reward', reward)

    def _experience(self, state, action, reward, successor):
        super()._experience(state, action, reward, successor)
        self._memory.append(state, action, reward, successor)
        if len(self._memory) < self._config.batch_size:
            return
        future = self._target.best(state=successor)
        future[np.equal(successor, None)] = 0
        target = reward + self._config.discount * future
        self._target.variables = self._actor.variables
        self._actor.train(state=previous, action=action, target=target)

    def _build_q_network(self, name):
        model = Model(name)
        with model.scope:
            model.placeholder('state', self._env.states.shape)
            model.placeholder('action_')
            model.placeholder('target')
            x = conv2d(model.state, 16, 8, 4, tf.nn.relu)
            x = conv2d(x, 32, 4, 3, tf.nn.relu)
            x = dense(x, 256, tf.nn.relu)
            x = dense(x, self._env.actions.shape, tf.nn.relu)
            cost = ((model.action_ * x) - model.target) ** 2
            model.action('best', tf.reduce_max(x, 1))
            model.action('choice', tf.argmax(x, 1))
            model.compile(cost, self._config.optimizer)
        return model

    @staticmethod
    def _default_config():
        discount = 0.99
        frame_skip = 4
        replay_capacity = int(1e4)
        batch_size = 32
        learning_rate = 1e-2
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        epsilon = AttrDict(start=0.3, stop=0.05, over=200)
        return AttrDict(**locals())


class ReplayMemory:

    def __init__(self, maxlen=None, seed=0):
        self._previous = collections.deque(maxlen=maxlen)
        self._action = collections.deque(maxlen=maxlen)
        self._reward = collections.deque(maxlen=maxlen)
        self._successor = collections.deque(maxlen=maxlen)
        self._random = np.random.RandomState(seed)
        self._length = 0

    def __len__(self):
        return self._length

    def append(self, previous, action, reward, successor):
        self._previous.append(previous)
        self._action.append(action)
        self._reward.append(reward)
        self._successor.append(successor)
        self._length += 1

    def sample(self, amount):
        previous = np.empty((amount,) + self._previous[0].shape)
        action = np.empty((amount,) + self._action[0].shape)
        reward = np.empty(amount)
        successor = np.empty((amount,) + self._successor[0].shape)
        choices = self._random.choice(len(self), amount, replace=False)
        for index, choice in enumerate(choices):
            previous[index] = self._previous[choice]
            action[index] = self._action[choice]
            reward[index] = self._reward[choice]
            successor[index] = self._successor[choice]
        return previous, action, reward, successor
