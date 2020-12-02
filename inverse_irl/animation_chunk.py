# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:41:37 2020

@author: Office
"""
def animate(episode):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow as tf
    
    import gym
    import gym_bicycle
    import os
    videos_path = 'D:/Hossam/GAIL Paper/videos'
    #for episode in range(21, 40):
    
#    episode = 8
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
