# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:47:30 2020

@author: Office
"""

import numpy as np
import gym
import gym_bicycle
import matplotlib.pyplot as plt
import pandas as pd
from gym import wrappers
import tensorflow as tf

env = gym.make('bicycle-v0')


env.reset()

env.render()

with tf.Session():    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    follow_actions = policy.get_action(observation= env.state)
    
env.step(action = follow_actions[0])
env.render()


env_to_wrap = gym.make('bicycle-v0')


env_to_wrap.reset()

#env_to_wrap.render()

env = wrappers.Monitor(env_to_wrap, 'D:/Hossam/GAIL Paper/videos', video_callable=lambda episode_id: True, force = True, mode=monitor_mode)
env.reset()
env.render()

env._start('D:/Hossam/GAIL Paper/videos')


for i in range(100):
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        follow_actions = policy.get_action(observation= env_to_wrap.state)

    
    state, reward, done, _ = env_to_wrap.step(follow_actions[0])
    
    
    env_to_wrap.render()
    env.render()
    print(i, state, done )
    
env_to_wrap.close()
  
env.close()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import gym
import gym_bicycle
import os
videos_path = 'D:/Hossam/GAIL Paper/videos'
for episode in range(10, 20):
    env = gym.make('bicycle-v0')
    episode_path = os.path.join(videos_path, str(episode))
    env = gym.wrappers.Monitor(env, episode_path, video_callable = lambda episode_id: True, force = True)
    #env._max_episode_steps = 399
    observation = env.reset()
    traj_val = observation

    for t in range(200):
        print(observation)
        with tf.Session():    
            sess = tf.get_default_session()
            sess.run(tf.global_variables_initializer())
            follow_actions = policy.get_action(observation= observation)
        observation, reward, done, info = env.step(follow_actions[0])
        traj_val1 = np.vstack([traj_val, observation])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    #    if (t > 300):
    #        print("Episode finished after {} timesteps".format(t+1))
    #        break
    base_traj_path = 'D:/Hossam/GAIL Paper/videos/video_trajectories/traj_vid_'
    traj_path = base_traj_path + str(episode) + '.csv'
    np.savetxt(traj_path, traj_val1, delimiter=",")
    env.close()
