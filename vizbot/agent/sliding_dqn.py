import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, fully_connected
from vizbot.core import Agent
from vizbot.utility import AttrDict, lazy_property


class Frames(Agent):

    def __init__(self, env, width):
        super().__init__(env)
        self._width = width
        self._frames = None

    def step(self, state):
        super().step(state)
        self._frames.append(state)
        if len(self._frames) < self._width:
            return self._noop()
        return self._step(np.array(self._frames))

    def begin(self):
        super().begin()
        self._frames = collections.deque(maxlen=self._width)

    def feedback(self, previous, action, reward, successor):
        super().feedback(previous, action, reward, successor)
        if len(self._frames):
            return
        previous = np.array(self._frames)
        successor = np.array(self._frames[1:] + [successor])
        self._feedback(previous, action, reward, successor)

    def _step(self, state):
        raise NotImplementedError

    def _feedback(self, previous, action, reward, successor):
        raise NotImplementedError


class SlidingDQN(Frames):

    def __init__(self, env, config=None):
        self._config = config or self._default_config()
        super().__init__(env, self._config.input_frames)
        self._random = np.random.RandomState(0)
        self._experience = ReplayMemory(self._config.replay_capacity)
        self._input = tf.placeholder(
            tf.float32, (None,) + self._states[: -1] + (self._config.input_frames,))
        self._action = tf.placeholder(tf.float32, (None, self._actions))
        self._target = tf.placeholder(tf.float32, (None,))
        self._prediction
        self._optimize
        self._sess = tf.Session()
        # TODO: Not good with multiple agents.
        self._sess.run(tf.initialize_all_variables())

    def _step(self, state):
        state = self._process_state(state)
        action = self._noop()
        epsilon = self._decay(**self._config.epsilon)
        if self._random.rand() < epsilon:
            choice = self._random.choice(self._actions)
        else:
            choice = self._sess.run(self._choice, {self._input: [state]})[0]
            print('DQN choice', choice)
        action[choice] = 1
        return action

    def _feedback(self, previous, action, reward, successor):
        previous = self._process_state(previous)
        successor = self._process_state(successor)
        self._experience.append(previous, action, reward, successor)
        if len(self._experience) < self._config.batch_size:
            return
        previous, action, reward, successor = \
            self._experience.sample(self._config.batch_size)
        feed = {self._input: [0 if x is None else x for x in successor]}
        future = self._sess.run(self._max, feed)
        future[np.equal(successor, None)] = 0
        target = reward + self._config.discount * future
        feed = {self._input: previous, self._action: action, self._target: target}
        self._sess.run(self._optimize, feed)

    @lazy_property
    def _prediction(self):
        x = self._input
        x = convolution2d(x, 16, 8, 4, 'VALID', tf.nn.relu)
        x = convolution2d(x, 32, 4, 2, 'VALID', tf.nn.relu)
        x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])
        x = fully_connected(x, 256, tf.nn.relu)
        output = fully_connected(x, self._actions)
        return output

    @lazy_property
    def _choice(self):
        return tf.argmax(self._prediction, 1)

    @lazy_property
    def _max(self):
        return tf.reduce_max(self._prediction, 1)

    @lazy_property
    def _optimize(self):
        prediction = tf.reduce_sum(self._prediction * self._action, 1)
        cost = (self._prediction - self._target) ** 2
        cost = tf.reduce_sum(cost)
        return self._config.optimizer.minimize(cost)

    @staticmethod
    def _process_state(state):
        state = np.transpose(state.mean(3), [1, 2, 0])
        return state

    @staticmethod
    def _default_config():
        discount = 0.95  # TODO: Find correct value in the paper.
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

