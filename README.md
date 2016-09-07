Mindpark
========

Testbed for deep reinforcement learning algorithms.

![DQN playing Breakout](http://imgur.com/zmwTvUx.gif)&nbsp;&nbsp;
![DQN playing Doom Health Gathering](http://imgur.com/ADsdHUM.gif)&nbsp;&nbsp;
![DQN trying to play Doom Deathmatch](http://imgur.com/WKDVGtx.gif)

## Instructions

Run an experiment to compare between algorithms, hyper parameters, and
environments:

```shell
python3 -O -m mindpark run definition/breakout.yaml
```

Videos and metrics are stored in a result directory, which is
`~/experiment/mindpark/<timestamp>-breakout/` by default. You can plot
statistics during or after the simulation by fuzzy matching an the folder name:

```shell
python3 -m mindpark stats breakout
```

![DQN statistics on Breakout](http://i.imgur.com/eh1K0Zl.png)

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

## Algorithms

- Deep Q-Network (Mnih et al. 2015, [PDF][dqn-paper])
- Asynchronous Advantage Actor-Critic (Mnih et al. 2016, [PDF][a3c-paper])

[dqn-paper]: https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
[a3c-paper]: https://arxiv.org/pdf/1602.01783v2.pdf

## Contribution

Pull requests are welcome. I will set up a contributors file then, and you can
choose if you want to be listed. Please follow the existing code style, and run
unit tests and the integration test after changes:

```shell
python setup.py test
python -m mindpark run definition/test.py -x
```

## Contact

Feel free to reach out if you have any questions.
