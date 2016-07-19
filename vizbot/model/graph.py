import glob
import os
import tensorflow as tf


class Graph:
    """
    Interface for a TensorFlow graph. Supports saving, loading, and restoring
    the whole graph from disk. Nodes are added by name and will be added to
    graph collections internally, so that they are available after loading.
    """

    def __init__(self):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._saver = None
        self._scope = None

    def __enter__(self):
        self._assert_modifiable()
        self._scope = self._graph.as_default()
        self._scope.__enter__()
        return self

    def __exit__(self, type_, value, traceback):
        self._saver = tf.train.Saver(self.variables)
        self._sess.run(tf.initialize_variables(self.variables))
        self._graph.finalize()
        self._scope.__exit__(type_, value, traceback)

    def load(self, directory):
        self._assert_modifiable()
        if not os.path.isdir(directory):
            raise ValueError('load directory does not exist')
        filepath = glob.glob(os.path.join(load_dir, '*.meta'))[0]
        self._saver = tf.train.import_meta_graph(filepath)
        self._saver.restore(self._sess, filepath[:-5])
        self._graph.finalize()

    def save(self, directory):
        self._assert_finalized()
        self._saver.save(
            self._sess, self._save_dir,
            latest_filename='checkpoint',
            meta_graph_suffix='meta',
            write_meta_graph=True)

    @property
    def variables(self):
        return self._graph.get_collection(tf.GraphKeys.VARIABLES)

    @property
    def weights(self):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    @property
    def weight_names(self):
        return {x.name for x in self.weights}

    def __contains__(self, name):
        return len(self._graph.get_collection(name)) > 0

    def __setitem__(self, name, node):
        self._assert_modifiable()
        self._assert_scope()
        if name in self:
            raise KeyError(name + ' already exists in the graph')
        self._graph.add_to_collection(name, node)

    def __getitem__(self, name):
        if name not in self:
            raise KeyError(name + ' does not exist in the graph')
        return self._graph.get_collection(name)[0]

    def __call__(self, ops, data=None):
        """
        Run one or more operations on the graph
        """
        self._assert_finalized()
        single = not isinstance(ops, (tuple, list))
        if single:
            ops = (ops,)
        if isinstance(ops, (list, tuple)):
            ops = [self[x] if isinstance(x, str) else x
                   for x in ops]
        data = data or {}
        data = {self[k] if isinstance(k, str) else k: v
                for k, v in data.items()}
        results = self._sess.run(ops, data)
        if single:
            results = results[0]
        return results

    def find(self, prefix):
        nodes = {}
        for key in self._graph._collections:
            if not isinstance(key, str):
                continue
            if not key.startswith(prefix):
                continue
            nodes[key[len(prefix):]] = self[key]
        return nodes

    def _assert_modifiable(self):
        if self._graph.finalized:
            raise RuntimeError('the graph cannot be modified anymore')

    def _assert_finalized(self):
        if not self._graph.finalized:
            raise RuntimeError('the graph is not ready for usage')

    def _assert_scope(self):
        if tf.get_default_graph() != self._graph:
            raise RuntimeError('tried to add operation from fron another')
