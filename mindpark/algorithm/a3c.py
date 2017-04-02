from threading import Lock
import numpy as np
import tensorflow as tf
import mindpark as mp


class A3C(mp.Algorithm):

    """
    Algorithm: Asynchronous Advantage Actor Critic (A3C)
    Paper: Asynchronous Methods for Deep Reinforcement Learning
    Authors: Mnih et al. 2016
    PDF: https://arxiv.org/pdf/1602.01783v2.pdf
    """

    @classmethod
    def defaults(cls):
        preprocess = 'dqn_2015'
        preprocess_config = dict()
        network = 'a3c_lstm'
        learners = 16
        approximation_config = dict(
            actor_weight=1.0, critic_weight=0.5, entropy_weight=0.01)
        apply_gradient = 5
        initial_learning_rate = 7e-4
        optimizer = tf.train.RMSPropOptimizer
        optimizer_config = dict(decay=0.99, epsilon=0.1)
        return mp.utility.merge_dicts(super().defaults(), locals())

    def __init__(self, task, config):
        super().__init__(task, config)
        self._preprocess = self._create_preprocess()
        self.model = mp.model.Model(self._create_network)
        # print(str(self.model))
        self.learning_rate = mp.utility.Decay(
            float(self.config.initial_learning_rate), 0, self.task.steps)
        self.lock = Lock()
        self.cost_metric = mp.Metric(self.task, 'a3c/cost', 1)
        self.value_metric = mp.Metric(self.task, 'a3c/value', 1)
        self.choice_metric = mp.Metric(self.task, 'a3c/choice', 1)

    @property
    def train_policies(self):
        trainers = []
        for _ in range(self.config.learners):
            config = mp.utility.AttrDict(self.config.copy())
            # TODO: Use single model to share RMSProp statistics. Does RMSProp
            # use statistics in compute_gradients() or apply_gradients()?
            model = mp.model.Model(self._create_network, threads=1)
            model.weights = self.model.weights
            policy = mp.Sequential(self.task)
            policy.add(self._create_preprocess())
            policy.add(Train, config, self, model)
            trainers.append(policy)
        return trainers

    @property
    def test_policy(self):
        policy = mp.Sequential(self.task)
        policy.add(self._preprocess)
        policy.add(Test, self.model, self)
        return policy

    def end_epoch(self):
        super().end_epoch()
        if self.task.directory:
            self.model.save(self.task.directory, 'model')

    def _create_network(self, model):
        learning_rate = model.add_option(
            'learning_rate', self.config.initial_learning_rate)
        model.set_optimizer(self.config.optimizer(
            learning_rate=learning_rate, use_locking=True,
            **self.config.optimizer_config))
        network = getattr(mp.part.network, self.config.network)
        observs = self._preprocess.above_task.observs.shape
        actions = self._preprocess.above_task.actions.n
        mp.part.approximation.advantage_policy_gradient(
            model, network, observs, actions, self.config.approximation_config)

    def _create_preprocess(self):
        policy = mp.Sequential(self.task)
        preprocess = getattr(mp.part.preprocess, self.config.preprocess)
        policy.add(preprocess, self.config.preprocess_config)
        return policy


class Train(mp.step.Experience):

    def __init__(self, task, config, master, model):
        super().__init__(task)
        self._config = config
        self._master = master
        self._model = model
        observ_shape = self.task.observs.shape
        shapes = (observ_shape, tuple(), tuple(), observ_shape)
        self._batch = mp.part.replay.Sequential(
            self._config.apply_gradient, shapes)
        self._context_last_batch = None

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._context_last_batch = None
        if self._model.has_option('context'):
            self._model.reset_option('context')

    def perform(self, observ):
        assert self.training
        choice, value = self._model.compute(
            ('choice', 'value'), state=observ)
        self._master.choice_metric(choice)
        self._master.value_metric(value)
        return choice

    def experience(self, observ, action, reward, successor):
        self._batch.push(observ, action, reward, successor)
        done = (successor is None)
        if not done and len(self._batch) < self._config.apply_gradient:
            return
        return_ = (
            0 if done else self._model.compute('value', state=observ))
        self._train(self._batch[:], return_)
        self._batch.clear()
        self._model.weights = self._master.model.weights

    def _train(self, transitions, return_):
        observs, actions, rewards, _ = transitions
        returns = self._compute_eligibilities(rewards, return_)
        self._decay_learning_rate()
        if self._model.has_option('context'):
            context = self._context_last_batch
            if context is not None:
                self._model.set_option('context', context)
            else:
                self._model.reset_option('context')
        delta, cost = self._model.delta(
            'cost', action=actions, state=observs, return_=returns)
        # self._log_gradient(delta)
        self._master.model.apply(delta)
        self._master.cost_metric(cost)
        if self._model.has_option('context'):
            self._context_last_batch = self._model.get_option('context')

    def _compute_eligibilities(self, rewards, return_):
        returns = []
        for reward in reversed(rewards):
            return_ = reward + self._config.discount * return_
            returns.append(return_)
        returns = np.array(list(reversed(returns)))
        return returns

    def _decay_learning_rate(self):
        learning_rate = self._master.learning_rate(self._master.task.step)
        self._master.model.set_option('learning_rate', learning_rate)

    # def _log_gradient(self, delta):
    #     keys = sorted(delta.keys())
    #     vals = [delta[x].mean() for x in keys]
    #     for key, val in zip(keys, vals):
    #         print('{:<40} {:20.16f}'.format(key, val))


class Test(mp.Policy):

    def __init__(self, task, model, master):
        super().__init__(task)
        self._model = model
        self._master = master

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        if self._model.has_option('context'):
            self._model.reset_option('context')

    def observe(self, observ):
        super().observe(observ)
        assert not self.training
        choice, value = self._model.compute(
            ('choice', 'value'), state=observ)
        self._master.choice_metric(choice)
        self._master.value_metric(value)
        return choice

    def receive(self, reward, final):
        super().receive(reward, final)
