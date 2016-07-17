Vizbot
======

Deep Reinforcement Learning agents and their evaluation.

## Instructions

Benchmark

```shell
python3 -m vizbot -d Env, ... -a Agent, ... -v -n 5e6
```

Recording

```shell
python3 -m vizbot -c -r 1 -a KeyboardDoom -n 10
```

## Tips

Monitoring

```
cd ~/experiment/gym && cd `ls -t | head -n 1`
watch "{ nvidia-smi; echo; ls; echo; cat *.stats.json; }"
```

## Setup

Dependencies

```shell
sudo -H pip3 install -U gym tensorflow
```
