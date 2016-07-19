import numpy as np
import tensorflow as tf
from vizbot.utility import AttrDict, lazy_property


class Model:

    def __init__(self, creator, optimizer=None):
        self._graph = tf.Graph()
        self._node = AttrDict(
            input_={}, output={}, cost={}, minimize={}, delta={})
        with self._graph.as_default():
            self._optimizer = optimizer or tf.train.RMSPropOptimizer(0.01)
            creator(self)
            self._create_apply_weight()
            self._create_apply_delta()
        self._sess = tf.Session(graph=self._graph)
        self._sess.run(tf.initialize_variables(self._all_variables))
        self._graph.finalize()

    def add_input(self, name, shape=None, type_=tf.float32):
        self._assert_scope()
        self._validate_input_name(name)
        shape = (shape,) if isinstance(shape, int) else shape
        shape = (None,) + (shape or tuple())
        input_ = tf.placeholder(type_, shape)
        self._node.input_[name] = input_
        return input_

    def add_output(self, name, tensor):
        self._assert_scope()
        self._validate_name(name)
        self._node.output[name] = tensor
        return tensor

    def add_cost(self, name, tensor):
        self._assert_scope()
        self._validate_name(name)
        cost = tf.reduce_sum(tensor)
        self._node.cost[name] = cost
        self._node.minimize[name] = self._optimizer.minimize(cost)
        vars_ = self._weight_variables
        grad_vars = self._optimizer.compute_gradients(cost, vars_)
        self._node.delta[name] = {var: gard for gard, var in grad_vars}
        return cost

    def train(self, cost, batch=None, epochs=1, **data):
        self._assert_finalized()
        keys = sorted(data.keys())
        data = [data[x] for x in keys]
        placeholders = [self._node.input_[x] for x in keys]
        batches = [self._chunks(x, batch) for x in data]
        costs = []
        for _ in range(epochs):
            for batch in zip(*batches):
                ops = self._node.cost[cost], self._node.minimize[cost]
                feed = {k: v for k, v in zip(placeholders, batch)}
                cost, _ = self._sess.run(ops, feed)
                costs.append(cost)
        return costs

    def compute(self, output, **data):
        self._assert_finalized()
        single = all(
            len(value.shape) + 1 == len(self._node.input_[key].get_shape())
            for key, value in data.items())
        if single:
            data = {k: np.expand_dims(v, 0) for k, v in data.items()}
        feed = {self._node.input_[k]: v for k, v in data.items()}
        result = self._sess.run(self._node.output[output], feed)
        if single:
            result = np.squeeze(result, 0)
        return result

    @property
    def weights(self):
        self._assert_finalized()
        vars_ = self._weight_variables
        values = self._sess.run(vars_)
        weights = {var.name: val for var, val in zip(vars_, values)}
        return weights

    @weights.setter
    def weights(self, weights):
        self._assert_finalized()
        if not all(name in self._weight_names for name in weights):
            raise KeyError('unrecognized weight name')
        feed = {self._apply_weight_ins[name]: value
                for name, value in weights.items()}
        self._sess.run(self._apply_weight, feed)

    def delta(self, cost, **data):
        self._assert_finalized()
        vars_ = self._weight_variables
        delta = [self._node.delta[cost][var] for var in vars_]
        values = self._sess.run(delta, data)
        delta = {var: val for var, val in zip(vars_, values)}
        return delta

    def apply_delta(self, delta):
        self._assert_finalized()
        if not all(name in self._weight_names for name in delta):
            raise KeyError('unrecognized weight name')
        feed = {self._apply_delta_ins[name]: value
                for name, value in delta.items()}
        print('Update', len(feed))
        self._sess.run(self._apply_delta, feed)

    def __str__(self):
        string = ''
        string += self._format_section('Inputs', self._node.input_)
        string += self._format_section('Outputs', self._node.output)
        string += self._format_section('Costs', self._node.cost)
        weights = {x.name: x for x in self._weight_variables}
        string += self._format_section('Weights', weights)
        string += '\n'
        return string

    @property
    def _all_variables(self):
        return self._graph.get_collection(tf.GraphKeys.VARIABLES)

    @property
    def _weight_variables(self):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    @lazy_property
    def _weight_names(self):
        self._assert_finalized()
        return {x.name for x in self._weight_variables}

    def _format_section(self, title, nodes):
        string = '\n{}:'.format(title)
        for name in sorted(nodes.keys()):
            shape = [str(x) for x in nodes[name].get_shape()]
            string += '\n  {} ({})'.format(name, ', '.join(shape))
        return string

    def _create_apply_weight(self):
        self._apply_weight_ins = {}
        for var in self._weight_variables:
            input_ = tf.placeholder_with_default(var, var.get_shape())
            self._apply_weight_ins[var.name] = input_
        assignments = [
            x.assign(self._apply_weight_ins[x.name])
            for x in self._weight_variables]
        self._apply_weight = assignments

    def _create_apply_delta(self):
        self._apply_delta_ins = {}
        for var in self._weight_variables:
            zeros = tf.zeros(var.get_shape(), tf.float32)
            input_ = tf.placeholder_with_default(zeros, var.get_shape())
            self._apply_delta_ins[var.name] = input_
        grad_vars = [
            (self._apply_delta_ins[x.name], x)
            for x in self._weight_variables]
        self._apply_delta = self._optimizer.apply_gradients(grad_vars)

    def _chunks(self, data, size=None):
        if not size:
            yield data
            return
        for index in range(0, len(data), size):
            yield data[index: index + size]

    def _assert_scope(self):
        if tf.get_default_graph() != self._graph:
            raise RuntimeError('cannot add model operations to another graph')

    def _validate_input_name(self, name):
        self._validate_name(name)
        if name in ('cost', 'output', 'batch', 'epochs'):
            raise KeyError(name + ' is an illegal input name')

    def _validate_name(self, name):
        if any(name in nodes for category, nodes in self._node.items()):
            raise KeyError(name + ' already exists')

    def _assert_finalized(self):
        if not self._graph.finalized:
            raise RuntimeError('graph must be finalized before using it')
