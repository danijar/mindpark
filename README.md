Mindpark
========

Testbed for deep reinforcement learning algorithms.

## Instructions

Run an experiment to compare algorithms, hyper parameters and environments:

```shell
python3 -O -m mindpark run definition/breakout.yaml
```

Videos and metrics are stored in a result directory. You can plot statistics
while running:

```shell
python3 -m mindpark stats ~/experiment/mindpark/<timestamp>-breakout
```

To implement your own algorithm, subclass `mindpark.core.Algorithm`. Please
refer to the existing algorithms for details, and ask if you have questions.

## Definitions

Definitions are YAML files that contain all you need to run or reproduce an
experiment:

```yaml
epochs: 100
test_steps: 4e4
repeats: 5
envs:
  - Pong-v0
  - Breakout-v0
algorithms:
  -
    name: LSTM-A3C (3 layers)
    type: A3C
    train_steps: 8e5
    config:
      network: lstm_three_layers
      initial_learning_rate: 2e-4
  -
    name: DQN (Mnih et al. 2015)
    type: DQN
    train_steps: 2e5
  -
    name: Random
    type: Random
    train_steps: 0
```

Each algorithm will be trained on each environment for the specified number of
repeats. A simulation is divided into epochs that consist of a training and an
evaluation phase.

## Arguments

```
usage: mindpark run [-h] [-o DIRECTORY] [-p PARALLEL] [-v VIDEOS] [-x]
                    definition

positional arguments:
  definition            YAML file describing the experiment

optional arguments:
  -h, --help            show this help message and exit
  -o DIRECTORY, --directory DIRECTORY
                        root folder for all experiments (default:
                        ~/experiment/mindpark)
  -p PARALLEL, --parallel PARALLEL
                        how many algorithms to train in parallel (default: 1)
  -v VIDEOS, --videos VIDEOS
                        how many videos to capture per epoch (default: 1)
  -x, --dry-run         do not store any results (default: False)
```

## Dependencies

Mindpark is a Python 3 package, and there are no plans to support Python 2. If
you run into problems, please install the dependencies manually via `pip3`:

- TensorFlow
- Gym
- PyYaml
- Matplotlib
- SQLAlchemy

TensorFlow is only needed for the existing algorithms. You are free to use your
libraries of choice to implement your own algorithms.

## Contribution

Pull requests are welcome. I will set up a contributors file then, and you can
choose if you want to be listed. Please follow the existing code style, and run
unit tests and integration test after changes:

```shell
python setup.py test
python -m mindpark run definition/test.py -x
```

## Contact

Feel free to reach out if you have any questions.
