Vizbot
======

Testbed for deep reinforcement learning algorithms.

## Instructions

```shell
python3 -O -m vizbot.bench definition/full.yaml -p <threads>
python3 -m vizbot.plot "~/experiment/gym/*"
```

## Development

```shell
python3 setup.py test --args -x
```

## Dependencies

- Python 3
- TensorFlow 0.9
- Gym
- PyYaml
- Matplotlib
