import collections
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

from vizbot.agent import EpsilonGreedy
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, lazy_property


class DQN(EpsilonGreedy):

    def __init__(self, env, config=None):
        self._config = config or self._default_config()
        env = Grayscale(env)
        env = Downsample(env, 4)
        env = FrameSkip(env, self._config.frame_skip)
        # env = OneHotActions(env)
        super().__init__(env, **self._config.epsilon)
        self._memory = ReplayMemory(self._config.replay_capacity)
        # K.set_session(tf.Session())
        self._actor = self._build_q_network(trainable=True)
        self._target = self._build_q_network(trainable=False)
        self._sync_target()

    def _perform(self, state):
        action = self._noop()
        q_values = self._actor.predict(np.array([state]))[0]
        choice = q_values.argmax()
        # print('Choice', choice)
        action[choice] = 1
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
        self._train(*self._memory.sample(self._config.batch_size))

    def _train(self, previous, action, reward, successor):
        future = self._target.predict(successor, action).max(1)
        future[np.equal(successor, None)] = 0
        target = reward + self._config.discount * future
        self._sync_target()
        # TODO: Only train output node of the real action.
        self._actor.train(previous, target)

    def _sync_target(self):
        self._target.set_weights(self._actor.get_weights())

    def _build_q_network(self, trainable):
        state = Input(shape=self._env.states.shape)
        x = Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu',
            trainable=trainable, dim_ordering='tf')(state)
        x = Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu',
            trainable=trainable, dim_ordering='tf')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', trainable=trainable)(x)
        q_values = Dense(self._env.actions.shape, activation='relu',
            trainable=trainable)(x)
        model = Model(input=state, output=q_values)
        model.compile(self._config.optimizer, 'mse')
        return model

    @staticmethod
    def _default_config():
        discount = 0.99
        frame_skip = 4
        replay_capacity = int(1e4)
        batch_size = 32
        optimizer = RMSprop(1e-2)
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
