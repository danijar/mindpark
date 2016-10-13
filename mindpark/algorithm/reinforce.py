import threading
import numpy as np
import tensorflow as tf
import mindpark as mp
import mindpark.part.preprocess
import mindpark.part.approximation
import mindpark.part.network


class Reinforce(mp.Algorithm):

    """
    Algorithm: Reinforce
    Paper: Simple Statistical Gradient-Following Algorithms for Connectionist
           Reinforcement Learning
    Author: Williams 1992
    PDF: http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    """

    @classmethod
    def defaults(cls):
        preprocess = 'default'
        preprocess_config = dict()
        network = 'dqn_2015'
        approximation = dict(scale_critic_loss=0.5, regularize=0.01)
        update_every = 10000
        batch_size = 32
        heads = 16
        initial_learning_rate = 2.5e-4
        optimizer = tf.train.RMSPropOptimizer
        optimizer_config = dict(decay=0.95, epsilon=0.1)
        return mp.utility.merge_dicts(super().defaults(), locals())

    def __init__(self, task, config):
        mp.Algorithm.__init__(self, task, config)
        self._preprocess = self._create_preprocess()
        self.model = mp.model.Model(self._create_network)
        print(str(self.model))
        self._learning_rate = mp.utility.Decay(
            self.config.initial_learning_rate, 0, self.task.steps)
        self.value_metric = mp.Metric(self.task, 'reinforce/value', 1)
        self.choice_metric = mp.Metric(self.task, 'reinforce/choice', 1)
        self._cost_metric = mp.Metric(self.task, 'reinforce/cost', 1)
        self._learning_rate_metric = mp.Metric(
            self.task, 'reinforce/learning_rate', 1)
        self._lock = threading.Lock()
        self._memory = self._create_memory()

    def end_epoch(self):
        super().end_epoch()
        if self.task.directory:
            self.model.save(self.task.directory, 'model')

    def head_submit_episode(self, episode):
        observs, actions, rewards, successors = zip(*episode)
        rewards = self._compute_eligibilities(rewards)
        with self._lock:
            for transition in zip(observs, actions, rewards, successors):
                self._memory.push(*transition)

    def head_maybe_update(self):
        if len(self._memory) < self.config.update_every:
            return
        with self._lock:
            self._decay_learning_rate()
            while len(self._memory) >= self.config.batch_size:
                batch = self._memory.batch(self.config.batch_size)
                self._train_network(batch)
            self._memory.clear()

    @property
    def train_policies(self):
        heads = []
        for _ in range(self.config.heads):
            policy = mp.Sequential(self.task)
            policy.add(self._create_preprocess())
            policy.add(Head, self)
            heads.append(policy)
        return heads

    @property
    def test_policy(self):
        policy = mp.Sequential(self.task)
        policy.add(self._preprocess)
        policy.add(Head, self)
        return policy

    def _train_network(self, batch):
        self.model.set_option(
            'learning_rate', self._learning_rate(self.task.step))
        self._learning_rate_metric(self.model.get_option('learning_rate'))
        observ, action, return_, _ = batch
        cost = self.model.train(
            'cost', state=observ, action=action, return_=return_)
        self._cost_metric(cost)

    def _create_preprocess(self):
        policy = mp.Sequential(self.task)
        preprocess = getattr(mp.part.preprocess, self.config.preprocess)
        preprocess(self.task, self.config)
        policy.add(preprocess, self.config.preprocess_config)
        return policy

    def _create_network(self, model):
        learning_rate = model.add_option(
            'learning_rate', self.config.initial_learning_rate)
        model.set_optimizer(self.config.optimizer(
            learning_rate=learning_rate,
            **self.config.optimizer_config))
        network = getattr(mp.part.network, self.config.network)
        observs = self._preprocess.above_task.observs.shape
        actions = self._preprocess.above_task.actions.n
        mp.part.approximation.value_policy_gradient(
            model, network, observs, actions, self.config.approximation)

    def _create_memory(self):
        observ_shape = self._preprocess.above_task.observs.shape
        shapes = observ_shape, tuple(), tuple(), observ_shape
        memory = mp.part.replay.Sequential(self.config.update_every, shapes)
        memory.log_memory_size()
        return memory

    def _compute_eligibilities(self, rewards):
        returns = []
        return_ = 0
        for reward in reversed(rewards):
            return_ = reward + self.config.discount * return_
            returns.append(return_)
        returns = np.array(list(reversed(returns)))
        return returns

    def _decay_learning_rate(self):
        learning_rate = self._learning_rate(self.task.step)
        self.model.set_option('learning_rate', learning_rate)


class Head(mp.step.Experience):

    def __init__(self, task, master):
        super().__init__(task)
        self._master = master
        self._episode = None

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._episode = []

    def end_episode(self):
        super().end_episode()
        if not self.task.training or not self._episode:
            return
        self._master.head_submit_episode(self._episode)
        self._master.head_maybe_update()

    def perform(self, observ):
        choice, value = self._master.model.compute(
            ('choice', 'value'), state=observ)
        self._master.choice_metric(choice)
        self._master.value_metric(value)
        return choice

    def experience(self, *transition):
        self._episode.append(transition)
