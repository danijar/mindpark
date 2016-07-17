import functools
import numpy as np
import tensorflow as tf


class Model:

    def __init__(self, name):
        self._placeholders = {}
        self._actions = {}
        self._graph = tf.get_default_graph()  # tf.Graph()
        self._name = name
        self._cost = None
        self._optimize = None
        self._sess = None

    @property
    def scope(self):
        # return self._graph.as_default()
        return tf.variable_scope(self._name)

    def placeholder(self, name, shape=None, type_=tf.float16):
        if hasattr(self, name) or name in ('batch_size', 'epochs'):
            raise KeyError('invalid placeholder name ' + name)
        shape = (None,) + shape if shape else (None,)
        placeholder = tf.placeholder(type_, shape)
        self._placeholders[name] = placeholder
        setattr(self, name, placeholder)

    def action(self, name, action):
        if hasattr(self, name):
            raise KeyError('invalid action name ' + name)
        self._actions[name] = action
        setattr(self, name, functools.partial(self._predict, action))

    def compile(self, cost, optimizer=None):
        with self.scope:
            self._cost = tf.reduce_sum(cost)
            optimizer = optimizer or tf.train.RMSPropOptimizer(0.01)
            self._optimize = optimizer.minimize(self._cost)
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    @property
    def variables(self):
        keys, variables = self._variables
        values = self._sess.run(variables)
        return keys, values

    @variables.setter
    def variables(self, keys_values):
        vars_ = {k: v for k, v in zip(*self._variables)}
        assignments = [
            tf.assign(vars_[k], v) for k, v in zip(*keys_values) if k in vars_]
        self._sess.run(assignments)

    def train(self, batch_size=None, epochs=1, **data):
        data = self._resolve_only_input(data)
        keys = sorted(data.keys())
        data = [data[x] for x in keys]
        placeholders = [self._placeholders[x] for x in keys]
        batches = [self._chunks(x, batch_size) for x in data]
        for _ in range(epochs):
            for batch in zip(*batches):
                feed = {k: v for k, v in zip(placeholders, batch)}
                self._sess.run(self._optimize, feed)

    @property
    def _variables(self):
        vars_ = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars_ = [var for var in vars_ if var.name.startswith(self._name)]
        keys = [var.name[len(self._name) + 1:] for var in vars_]
        return keys, vars_

    def _predict(self, output, **data):
        single = all(
            len(value.shape) + 1 == len(self._placeholders[key].get_shape())
            for key, value in data.items())
        if single:
            data = {k: np.expand_dims(v, 0) for k, v in data.items()}
        feed = {self._placeholders[k]: v for k, v in data.items()}
        result = self._sess.run(output, feed)
        if single:
            result = np.squeeze(result, 0)
        return result

    def _chunks(self, data, size):
        for index in range(0, len(data), size):
            yield data[index: index + size]
