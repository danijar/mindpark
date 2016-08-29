import glob
import os
from mindpark.run import Definition
from test.fixtures import *


class TestDefinition:

    def test_existing_definitions_valid(self):
        path = os.path.join(os.path.dirname(__file__), '../definition')
        path = os.path.abspath(path)
        for definition in glob.glob(os.path.join(path, '*.yaml')):
            print(definition)
            Definition(definition)

    def test_override_algorithm_config(self, algo_cls, task, algo_config):
        assert algo_cls(task, algo_config).config.discount != 0.42
        algo_config.discount = 0.42
        assert algo_cls(task, algo_config).config.discount == 0.42
