Vizbot
======

Testbed for deep reinforcement learning algorithms.

## Instructions

```shell
python3 -O -m mindpark.run definition/full.yaml
python3 -m mindpark.plot
```

The default result directory is `~/experiment/gym/<timestamp>-<experiment>/`.
It is used to store hyper parameters, evaluation scores, and videos.

## Options

```
usage: mindpark [-h] [-o DIRECTORY] [-p PARALLEL] [-x] [-v VIDEOS] definition

positional arguments:
  definition            YAML file describing the experiment

optional arguments:
  -h, --help            show this help message and exit
  -o DIRECTORY, --directory DIRECTORY
                        root folder for all experiments (default:
                        ~/experiment/gym)
  -p PARALLEL, --parallel PARALLEL
                        how many algorithms to train in parallel (default: 1)
  -x, --dry-run         do not store any results (default: False)
  -v VIDEOS, --videos VIDEOS
                        how many videos to capture per epoch (default: 1)
```

## Definitions

This is how you describe an experiment that compares different algorithms,
hyper parameters, and environments. All algorithms will be trained and
evaluated on all environments. For example:

```yaml
experiment: my-experiment
epochs: 100
test_steps: 4e4
repeats: 5
envs:
  - Pong-v0
  - Breakout-v0
algorithms:
  -
    name: LSTM-A3C (My custom network)
    type: A3C
    train_steps: 8e5
    network: lstm_three_layers
    initial_learning_rate: 7e-4
  -
    name: DQN (Mnih et al. 2015)
    type: DQN
    train_steps: 2e5
  -
    name: Random
    type: Random
    train_steps: 0
```

You don't need to provide default values. The full configuration of each
algorithm will be stored along with its results.

## Dependencies

TensorFlow is only a dependency because I chose it to implement the existing
algorithms. When implementing your own algorithm, you are free to use your
libraries of choice.

- Python 3
- TensorFlow
- Gym
- PyYaml
- Matplotlib
- SQLAlchemy

## Development

Run unit tests and integration test after changes.

```shell
python setup.py test
python -m mindpark.run definition/test.py -x
```

## Contact

Feel free to reach out if you have any questions.
