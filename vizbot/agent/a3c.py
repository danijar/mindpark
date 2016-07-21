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


class A3C(Agent):

    @classmethod
    def _config(cls):
        # Preprocess.
        downsample = 2
        frame_skip = 4
        # Heads.
        heads = 32
        epsilon = [
            AttrDict(start=1, stop=0.10, over=4e6),
            AttrDict(start=1, stop=0.01, over=4e6),
            AttrDict(start=1, stop=0.50, over=4e6)]
        # Learning.
        discount = 0.95
        apply_gradient = 5
        regularize = 0.01
        # optimizer = (tf.train.RMSPropOptimizer, 1e-5, 0.99)
        optimizer = (tf.train.RMSPropOptimizer, 0.1, 0.99)
        # External.
        save_model = int(1e5)
        print_cost = int(5e4)
        load_dir = ''
        return locals()

    def __init__(self, trainer, **config):
        self._config = AttrDict(merge_dicts(self._config(), config))
        trainer.add_preprocess(Grayscale)
        trainer.add_preprocess(Downsample, self._config.downsample)
        trainer.add_preprocess(FrameSkip, self._config.frame_skip)
        super().__init__(trainer)
        self.model = Model(self._create_network, self._config.optimizer)
        self.actor_costs = []
        self.critic_costs = []
        self._threads = self._create_threads()
        print(str(self.model))

    def __call__(self):
        for thread in self._threads:
            thread.start()
        save_model = Every(self._config.save_model)
        print_cost = Every(self._config.print_cost)
        while self._trainer.running:
            if print_cost(self._trainer.timestep):
                self._print_cost()
            if save_model(self._trainer.timestep) and self._trainer.directory:
                self.model.save(self._trainer.directory, 'model')
            time.sleep(0.01)
        for thread in self._threads:
            thread.join()

    def _create_network(self, model):
        # Network.
        state = model.add_input('state', self.states.shape)
        hidden = default_network(state)
        value = model.add_output('value',
            tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(hidden, self.actions.shape, tf.nn.softmax)
        sample = tf.squeeze(tf.multinomial(policy, 1), [1])
        model.add_output('choice', tf.one_hot(sample, self.actions.shape))
        # Training.
        action = model.add_input('action', self.actions.shape)
        return_ = model.add_input('return_')
        advantage = return_ - value
        gradient = tf.reduce_sum(tf.log(policy * action + 1e-9)) * advantage
        entropy = tf.reduce_sum(tf.log(policy + 1e-9) * policy)
        model.add_cost('actor',
            (gradient + self._config.regularize * entropy) ** 2)
        model.add_cost('critic', (return_ - value) ** 2)

    def _create_threads(self):
        threads = []
        for index in range(self._config.heads):
            agent = Head(self._trainer, self, **self._config)
            thread = Thread(None, agent, 'head-{}'.format(index))
            threads.append(thread)
        return threads

    def _print_cost(self):
        if not (self.actor_costs and self.critic_costs):
            return
        actor = sum(self.actor_costs) / len(self.actor_costs)
        critic = sum(self.critic_costs) / len(self.critic_costs)
        self.actor_costs = []
        self.critic_costs = []
        print('Cost actor {:12.2f} critic {:12.8f}'.format(actor, critic))


class Head(EpsilonGreedy):

    def __init__(self, trainer, master, **config):
        self._config = config
        epsilon = master._random.choice(config.epsilon)
        super().__init__(trainer, **epsilon)
        self._master = master
        self._batch = Experience(self._config.apply_gradient)

    def _step(self, state):
        return self._estimate('choice', state=state)

    def experience(self, state, action, reward, successor):
        self._batch.append((state, action, reward, successor))
        done = (successor is None)
        if not done and len(self._batch) < self._config.apply_gradient:
            return
        return_ = 0 if done else self._estimate('value', state=state)
        self._learn(self._batch.access(), return_)
        self._batch.clear()

    def _learn(self, transitions, return_):
        states, actions, rewards, _ = transitions
        returns = []
        for reward in reversed(rewards):
            return_ = reward + self._config.discount * return_
            returns.append(return_)
        returns = np.array(list(reversed(returns)))
        try:
            actor_cost = self._master.model.train('actor',
                action=actions, state=states, return_=returns)
            self._master.actor_costs.append(actor_cost)
        except DivergedError:
            print('Actor diverged')
        try:
            critic_cost = self._master.model.train('critic',
                state=states, return_=returns)
            self._master.critic_costs.append(critic_cost)
        except DivergedError:
            print('Critic diverged')

    def _estimate(self, output, **data):
        return self._master.model.compute(output, **data)
