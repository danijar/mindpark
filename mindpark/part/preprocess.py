import mindpark as mp


def default(task, config):
    defaults = dict(
        subsample=2, frame_skip=4, history=4, delta=False, frame_max=2,
        noop_max=30)
    config = mp.utility.use_attrdicts(mp.utility.merge_dicts(defaults, config))
    policy = mp.Sequential(task)
    policy.add(mp.step.Image)
    if config.noop_max:
        policy.add(mp.step.RandomStart, config.noop_max)
    if config.frame_skip > 1:
        policy.add(mp.step.Skip, config.frame_skip)
    if config.frame_max:
        policy.add(mp.step.Maximum, config.frame_max)
    if config.history > 1:
        channels = policy.above_task.observs.shape[-1]
        policy.add(mp.step.Grayscale, (0.299, 0.587, 0.114)[:channels])
    if config.subsample > 1:
        sub = config.subsample
        amount = (sub, sub) if config.history > 1 else (sub, sub, 1)
        policy.add(mp.step.Subsample, amount)
    if config.delta:
        policy.add(mp.step.Delta)
    if config.history > 1:
        policy.add(mp.step.History, config.history)
    policy.add(mp.step.Normalize)
    policy.add(mp.step.ClampReward)
    return policy


def dqn_2015(task, config=None):
    policy = mp.Sequential(task)
    policy.add(mp.step.Image)
    policy.add(mp.step.RandomStart, 30)
    policy.add(mp.step.Skip, 4)
    policy.add(mp.step.Maximum, 2)
    policy.add(mp.step.Grayscale, (0.299, 0.587, 0.114))
    policy.add(mp.step.Subsample, (2, 2))
    policy.add(mp.step.History, 4)
    policy.add(mp.step.Normalize)
    policy.add(mp.step.ClampReward)
    return policy

