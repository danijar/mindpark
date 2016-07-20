Vizbot
======

Testbed for deep deinforcement learning agents.

## Instructions

Benchmark

```shell
python3 -m vizbot -a DQN Q SARSA Random -l benchmark
```

Quick check

```shell
python3 -m vizbot -a SARSA Q DQN Random -n 3000 -l check -e 1000 -r 2
```

Recording

```shell
python3 -m vizbot -c -r 1 -a KeyboardDoom -n 10
```

## Setup

Dependencies

```shell
sudo -H pip3 install -U gym tensorflow
```
