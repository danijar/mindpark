import numpy as np
import tensorflow as tf
from vizbot.model.graph import Graph


class Model:
    """
    A flexible interface to define TensorFlow models. Similar to the Keras
    functional API but allows to directly add TensorFlow nodes, have multiple
    cost functions, compute gradients for costs, and apply gradients.
    """

    def __init__(self, creator=None, optimizer=None, load_path=None):
        """
        Create a new model. Either load_path or creator must be specified.

        Args:
            load_path (path, optional): Load a stored TensorFlow model.
            creator (callable, optional): Function that defines creates model
                if load_path is not specified. Will be executed with the graph
                of the model as default graph. After this function, no further
                operations can be added to the graph.
            optimizer (tuple, optional): A tuple of a TensorFlow optimizer
                class and its positional arguments.
        """
        optimizer = optimizer or (tf.train.RMSPropOptimizer, 0.01)
        self._graph = Graph()
        if load_path:
            try:
                self._graph.load(load_path)
                print('Loaded model')
                return
            except IOError:
                pass
        print('Create model')
        with self._graph:
            self._optimizer = optimizer[0](*optimizer[1:])
            creator(self)
            self._create_set_weight()
            self._create_apply_delta()

    def save(self, save_path):
        self._graph.save(save_path)

    def add_input(self, name, shape=None, type_=tf.float32):
        if name in ('cost', 'output', 'batch', 'epochs'):
            raise KeyError(name + ' is an illegal input name')
        shape = (shape,) if isinstance(shape, int) else shape
        shape = (None,) + (shape or tuple())
        node = tf.placeholder(type_, shape)
        self._graph['input/' + name] = node
        return node

    def add_output(self, name, node):
        self._graph['output/' + name] = node
        return node

    def add_cost(self, name, node):
        node = tf.reduce_sum(node)
        self._graph['cost/' + name] = node
        self._graph['optimize/' + name] = self._optimizer.minimize(node)
        return node

    def train(self, cost, batch=None, epochs=1, **data):
        data, _ = self._prepare_data(data)
        costs = []
        ops = 'cost/' + cost, 'optimize/' + cost
        for batch in self._chunks(data, batch, epochs):
            cost, _ = self._graph(ops, batch)
            costs.append(cost)
        return sum(costs) / len(costs)

    def compute(self, output, **data):
        data, single = self._prepare_data(data)
        result = self._graph('output/' + output, data)
        if single:
            result = np.squeeze(result, 0)
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
        values = self._graph('delta/' + cost, self._prepare_data(data)[0])
        return {var.name: val for var, val in zip(self.weights, values)}

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
