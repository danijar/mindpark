import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, fully_connected
from vizbot.core import Agent, Mixin, Add
from vizbot.mixin import Image, FrameSkip
from vizbot.utility import AttrDict, lazy_property


@Add(Image, 'grayscale', 2)
@Add(FrameSkip, 4)

class DQN(Agent):

    def __init__(self, actions, states, config=None):
        super().__init__(actions, states)
        self._config = config or self._default_config()
        self._memory = ReplayMemory(self._config.replay_capacity)
        self._input = tf.placeholder(tf.float32,
            (None,) + self._states.shape[: -1] + (self._config.input_frames,))
        self._action = tf.placeholder(tf.float32, (None, self._actions.shape))
        self._target = tf.placeholder(tf.float32, (None,))
        self._optimize
        self._sess = tf.Session()
        # TODO: Not good with multiple agents.
        self._sess.run(tf.initialize_all_variables())

    def step(self, state):
        super().step(state)
        epsilon = self._decay(**self._config.epsilon)
        if self._random.rand() < epsilon:
            choice = self._random.choice(self._actions.shape)
        else:
            choice = self._sess.run(self._predict, {self._input: [state]})
            choice = np.argmax(choice[0])
            print('DQN action', choice)
        action = self._noop()
        action[choice] = 1
        return action

    def feedback(self, action, reward):
        super().feedback(action, reward)
        if reward:
            print('DQN reward', reward)

    def _experience(self, state, action, reward, successor):
        self._memory.append(state, action, reward, successor)
        if len(self._memory) < self._config.batch_size:
            return
        previous, action, reward, successor = \
            self._memory.sample(self._config.batch_size)
        feed = {self._input: [0 if x is None else x for x in successor]}
        future = self._sess.run(self._predict, feed).max(1)
        future[np.equal(successor, None)] = 0
        target = reward + self._config.discount * future
        feed = {self._input: previous, self._action: action, self._target: target}
        self._sess.run(self._optimize, feed)

    @lazy_property
    def _predict(self):
        x = self._input
        x = convolution2d(x, 16, 8, 4, 'VALID', tf.nn.relu)
        x = convolution2d(x, 32, 4, 2, 'VALID', tf.nn.relu)
        x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])
        x = fully_connected(x, 256, tf.nn.relu)
        output = fully_connected(x, self._actions.shape)
        return output

    @lazy_property
    def _optimize(self):
        prediction = tf.reduce_sum(self._predict * self._action, 1)
        cost = (self._predict - self._target) ** 2
        cost = tf.reduce_sum(cost)
        return self._config.optimizer.minimize(cost)

    @staticmethod
    def _default_config():
        # TODO: Find correct discount factor in the paper.
        discount = 0.95
        input_frames=4
        replay_capacity = int(1e6)
        batch_size = 32
        learning_rate = 1e4
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        epsilon = AttrDict(start=1, end=0.1, over=int(1e6))
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
