Mindpark
========

Testbed for deep reinforcement learning algorithms.

![DQN playing Breakout](http://imgur.com/zmwTvUx.gif)&nbsp;&nbsp;
![DQN playing Doom Health Gathering](http://imgur.com/ADsdHUM.gif)&nbsp;&nbsp;
![DQN trying to play Doom Deathmatch](http://imgur.com/WKDVGtx.gif)

## Introduction

Reinforcement learning is a fundamental problem in artificial intelligence. In
this setting, an agent interacts with an environment in order to maximize a
reward. For example, we show our bot pixel screens of a game and want it to
choose actions that result in a high score.

Mindpark is an environment for prototyping, testing, and comparing algorithms
that do reinforcement learning. The library makes it easy to reuse part of
behavior between algorithms, and monitor all kinds of metrics about your
algorithms. It integrates well with TensorFlow, Theano, and other deep learning
libraries, and with OpenAI's gym environments.

These are the algorithms that I implemented so far (feel free to contribute to
this list):

- Deep Q-Network (Mnih et al. 2015, [PDF][dqn-paper])
- Asynchronous Advantage Actor-Critic (Mnih et al. 2016, [PDF][a3c-paper])

[dqn-paper]: https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
[a3c-paper]: https://arxiv.org/pdf/1602.01783v2.pdf

## Instructions

To get started, clone the repository and navigate into it:

```sh
git clone git@github.com:danijar/mindpark.git && cd mindpark
```

An experiment compares between algorithms, hyper parameters, and environments.
To start an experiment, run (`-O` turns on Python's optimizations):

```sh
python3 -O -m mindpark run definition/breakout.yaml
```

Videos and metrics are stored in a result directory, which is
`~/experiment/mindpark/<timestamp>-breakout/` by default. You can plot
statistics during or after the simulation by fuzzy matching an the folder name:

```sh
python3 -m mindpark stats breakout
```

## Statistics

Let's take a look at what the previous command creates.

Experiments consist of interleaved phases of training and evaluation. For
example, an algorithm might use a lower exploration rate in favor of
exploitation while being evaluated. Therefore, we display the metrics in two
rows:

![DQN statistics on Breakout](http://i.imgur.com/eh1K0Zl.png)

This illustrates the metrics after a few episodes of training, as you can see
on the horizontal axes. This small example is good for explanation. But if you
want to take a look, here are the [metrics of a longer
experiment][metrics-long].

| Metric | Description |
| ------ | ----------- |
| `score` | During the first 80 episodes of training (the time when I ran `mindpark stats`), the algorithm manages to get a score of 9, but usually get scores around 3 and 4. Below is the score during evaluation. It's lower because the algorithm hasn't learned much yet and performs worse than the random exploration done during training. |
| `dqn/cost` | The training cost of the neural network. It starts at episode 10 which is when the training starts, before that, DQN builds up its replay memory. We don't train the neural network during evaluation, so that plot is empty. |
| `epsilion_greedy/values` | That's the Q-values that the dqn behavior sends to epsilon_greedy to act greedily on. You can see that they evolve over time: Action 4 seems to be quite good. But that's only for a short run, so we shouldn't conclude too much. |
| `epsilion_greedy/random` | A histogram whether the current action was chosen randomly or greedy wrt the predicted Q-values. During training, epsilon is annealed, so you see a shift in the distribution. During testing, epsilon is always 0.05, so not many random actions there. |

The metric names are prefixed by the classes they come from. That's because
algorithms are composed of reusable partial behaviors. See the [Algorithms
section](#algorithms) for details.

[metrics-long]: http://i.imgur.com/qZBR7nB.png

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

## Algorithms

To implement your own algorithm, subclass `mindpark.core.Algorithm`. Please
refer to the existing algorithms for details, and ask if you have questions.
Algorithms are composed of partial behaviors that can do preprocessing,
exploration, learning, and more. To create a reusable chunk of behavior,
subclass `mindpark.core.Partial`.

There are quite a few existing behaviors that you can import from
`mindpark.step` and reuse in your algorithms. For more details, please look at
the according Python files or open an issue. Current behaviors include:
`ActionMax`, `ActionSample`, `ClampReward`, `Delta`, `EpsilonGreedy`,
`Experience`, `Filter`, `Grayscale`, `History`, `Identity`, `Maximum`,
`Normalize`, `Random`, `RandomStart`, `Resize`, `Score`, `Skip`, `Subsample`.

## Dependencies

Mindpark is a Python 3 package, and there are no plans to support Python 2 from
my side. If you run into problems, please manually install the dependencies via
`pip3`:

- TensorFlow
- Gym
- PyYaml
- Matplotlib
- SQLAlchemy

TensorFlow is only needed for the existing algorithms. You are free to use your
libraries of choice to implement your own algorithms.

## Contributions

Your pull request is very welcome. I will set up a contributors file in that
case, and you can choose if and how you want to be listed.

Please follow the existing code style, and run unit tests and the integration
test after changes:

```sh
python3 setup.py test
python3 -m mindpark run definition/test.yaml -x
```

## Contact

Feel free to reach out at [mail@danijar.com](mailto:mail@danijar.com) or open
an issue here on Github if you have any questions.
