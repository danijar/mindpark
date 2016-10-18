import os
import numpy as np
import tensorflow as tf
from mindpark.model.graph import Graph


class Model:
    """
    A flexible interface to define TensorFlow models. Similar to the Keras
    functional API but allows to directly add TensorFlow nodes, have multiple
    cost functions, compute gradients for costs, and apply gradients.
    """

    def __init__(
            self, creator=None, load_path=None, threads=None, clip_delta=10):
        """
        Create a new model. Either load_path or creator must be specified.

        Args:
            load_path (path, optional): Load a stored TensorFlow model.
            creator (callable, optional): Function that defines creates model
                if load_path is not specified. Will be executed with the graph
                of the model as default graph. After this function, no further
                operations can be added to the graph.
        """
        self._clip_delta = clip_delta
        self._graph = Graph(threads)
        self._optimizer = None
        if load_path:
            try:
                self._graph.load(load_path)
                return
            except IOError:
                pass
        with self._graph:
            creator(self)
            self._create_set_weight()
            self._create_apply_delta()

    def save(self, *path):
        self._graph.save(os.path.join(*path))

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def add_input(self, name, shape=None, type_=tf.float32):
        if name in ('cost', 'output', 'batch', 'epochs'):
            raise KeyError(name + ' is an illegal input name')
        shape = (shape,) if isinstance(shape, int) else shape
        shape = (None,) + (shape or tuple())
        node = tf.placeholder(type_, shape)
        self._graph['input/' + name] = node
        return node

    def has_input(self, name):
        return 'input/' + name in self._graph

    def add_option(self, name, initial, type_=tf.float32):
        node = tf.Variable(initial, trainable=False, dtype=type_)
        initial = tf.Variable(initial, trainable=False, dtype=type_)
        input_ = tf.placeholder_with_default(initial, initial.get_shape())
        self._graph['option/' + name] = node
        self._graph['option_input/' + name] = input_
        self._graph['option_set/' + name] = node.assign(input_)
        return node

    def has_option(self, name):
        return 'option/' + name in self._graph

    def get_option(self, name):
        return self._graph('option/' + name)

    def set_option(self, name, value):
        self._graph('option_set/' + name, {'option_input/' + name: value})

    def reset_option(self, name):
        self._graph('option_set/' + name)

    def add_output(self, name, node):
        self._graph['output/' + name] = node
        return node

    def has_output(self, name):
        return 'output/' + name in self._graph

    def add_cost(self, name, node):
        if not self._optimizer:
            raise RuntimeError('must set an optimizer before adding a cost')
        node = tf.reduce_sum(node)
        self._graph['cost/' + name] = node
        clip = self._clip_delta
        for gradient, variable in self._optimizer.compute_gradients(node):
            if gradient is None:
                continue
            clipped = tf.clip_by_value(gradient, -clip, clip)
            self._graph['delta/' + name + '/' + variable.name] = clipped
        return node

    def has_cost(self, name):
        return 'cost/' + name in self._graph

    def train(self, cost, batch=None, epochs=1, **data):
        costs = []
        for batch in self._chunks(data, batch, epochs):
            # TODO: See if training directly is more efficient.
            delta, cost = self.delta(cost, **data)
            self.apply(delta)
            costs.append(cost)
        return sum(costs) / len(costs)

    def compute(self, output, **data):
        data, single = self._prepare_data(data)
        single_out = not isinstance(output, (tuple, list))
        if single_out:
            output = (output,)
        result = self._graph(['output/' + x for x in output], data)
        if single:
            result = [np.squeeze(x, 0) for x in result]
        if single_out:
            result = result[0]
        return result

    @property
    def weights(self):
        values = self._graph(self._graph.weights)
        weight_vals = zip(self._graph.weights, values)
        weights = {var.name: val for var, val in weight_vals}
        return weights

    @weights.setter
    def weights(self, weights):
        self._validate_weights(weights)
        feed = {'set_weight_input/' + name: value
                for name, value in weights.items()}
        self._graph(['set_weight/' + x for x in weights], feed)

    def delta(self, cost, **data):
        data, _ = self._prepare_data(data)
        delta_nodes = list(self._graph.find('delta/' + cost + '/').items())
        delta_names = [x[0] for x in delta_nodes]
        ops = ['cost/' + cost] + [x[1] for x in delta_nodes]
        results = self._graph(ops, data)
        cost, delta = results[0], results[1:]
        if not np.isfinite(cost):
            print('the cost measure diverged')
        delta = dict(zip(delta_names, delta))
        return delta, cost

    def apply(self, delta):
        self._validate_weights(delta)
        feed = {'apply_delta_input/' + name: value
                for name, value in delta.items()}
        self._graph('apply_delta', feed)

    def _create_set_weight(self):
        for var in self._graph.weights:
            input_ = tf.placeholder_with_default(var, var.get_shape())
            self._graph['set_weight_input/' + var.name] = input_
            self._graph['set_weight/' + var.name] = var.assign(input_)

    def _create_apply_delta(self):
        for var in self._graph.weights:
            input_ = tf.placeholder_with_default(var, var.get_shape())
            self._graph['apply_delta_input/' + var.name] = input_
        grad_vars = [
            (self._graph['apply_delta_input/' + var.name], var)
            for var in self._graph.weights]
        self._graph['apply_delta'] = self._optimizer.apply_gradients(grad_vars)

    def _validate_weights(self, weights):
        for name in weights:
            if name not in self._graph.weight_names:
                raise KeyError('unrecognized weight name ' + name)

    def _prepare_data(self, data):
        data = {k: np.array(v) for k, v in data.items()}
        for name, values in data.items():
            if not np.isfinite(values).all():
                raise ValueError('non finite values in training input ' + name)
        single = all(
            len(val.shape) + 1 == len(self._graph['input/' + key].get_shape())
            for key, val in data.items())
        data = {'input/' + k: v for k, v in data.items()}
        if single:
            data = {k: np.expand_dims(v, 0) for k, v in data.items()}
        return data, single

    def _chunks(self, data, size=None, epochs=1):
        for _ in range(epochs):
            if not size:
                yield data
                continue
            for index in range(0, len(data), size):
                yield {k: v[index: index + size] for k, v in data}

    def __str__(self):
        string = ''
        string += self._format_section('Inputs', self._graph.find('input/'))
        string += self._format_section('Outputs', self._graph.find('output/'))
        string += self._format_section('Costs', self._graph.find('cost/'))
        weights = {x.name: x for x in self._graph.weights}
        string += self._format_section('Weights', weights)
        string += '\n'
        return string

    def _format_section(self, title, nodes):
        string = '\n{}:'.format(title)
        for name in sorted(nodes.keys()):
            shape = [str(x) for x in nodes[name].get_shape()]
            string += '\n  {} ({})'.format(name, ', '.join(shape))
        return string
