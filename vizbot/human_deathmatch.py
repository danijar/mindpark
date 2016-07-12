import numpy as np
import gym
from doom_py import ScreenResolution


env = gym.make('DoomDeathmatch-v0')
env.configure(screen_resolution=ScreenResolution.RES_160X120)
env.mode = 'human'

while True:
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action = np.zeros(env.action_space.shape, dtype=int)
        observation, reward, done, info = env.step(action)
