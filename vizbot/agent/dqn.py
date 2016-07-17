import collections
import numpy as np
import tensorflow as tf
from vizbot.core import Model
from vizbot.agent import EpsilonGreedy
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, lazy_property, dense, conv2d


class DQN(EpsilonGreedy):

    def __init__(self, env, config=None):
        self._config = config or self._default_config()
        env = Grayscale(env)
        env = Downsample(env, self._config.downsample)
        env = FrameSkip(env, self._config.frame_skip)
        super().__init__(env, **self._config.epsilon)
        self._memory = ReplayMemory(self._config.replay_capacity)
        self._actor = self._build_q_network()
        self._target = self._build_q_network()
        self._target.variables = self._actor.variables

    def _perform(self, state):
        return self._actor.perform(state=state)

    def _experience(self, state, action, reward, successor):
        self._memory.append(state, action, reward, successor)
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

    def _build_q_network(self):
        with Model() as model:
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
        # replay_capacity = int(1e6)
        replay_capacity = int(1e5)
        batch_size = 32
        learning_rate = 1e-4
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # epsilon = AttrDict(start=1, stop=0.1, over=int(1e6))
        epsilon = AttrDict(start=0.8, stop=0, over=int(1e5))
        return AttrDict(**locals())


class ReplayMemory:

    def __init__(self, maxlen=None):
        self._previous = collections.deque(maxlen=maxlen)
        self._action = collections.deque(maxlen=maxlen)
        self._reward = collections.deque(maxlen=maxlen)
        self._successor = collections.deque(maxlen=maxlen)
        self._random = np.random.RandomState(seed=0)
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
