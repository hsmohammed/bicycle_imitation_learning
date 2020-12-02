# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:43:20 2019

@author: Office
"""
import gym
import gym_bicycle

env = gym.make("bicycle-v0")
env.reset()
env.render()


for i in range(len(expert_training)):
    env.state = expert_training[i]['observations'][0]
    for j in range(len(expert_training[i]['observations'])):
        env.step(expert_training[i]['actions'][j])
        env.render()


env.close()

import gym
import gym_bicycle
import tensorflow as tf

env = gym.make("bicycle-v0")
env.reset()
env.render()

with tf.Session():
    tf.global_variables_initializer().run()
    for i in range(len(expert_training)):
        env.state = expert_training[i]['observations'][0]
        state = env.state
        while True:
            new_state, reward, done, _ = env.step(algo.policy.get_action(state)[0])
            state = new_state
            env.render()
            if done:
                break

import random            
random.seed(1000)           
with tf.Session():
    tf.global_variables_initializer().run()
    for i in {37,39, 40}:
        env.state = expert_training[i]['observations'][0]
        state = env.state
        while True:
            new_state, reward, done, _ = env.step(algo.policy.get_action(state)[0])
            state = new_state
            env.render()
            if done:
                break
   
env.close()

rewards = []
import random            
random.seed(1000)           
with tf.Session():
    tf.global_variables_initializer().run()
    for i in {37}:
        env.state = expert_training[i]['observations'][0]
        state = env.state
        while True:
            new_state, reward, done, _ = env.step(algo.policy.get_action(state)[0])
            state = new_state
            env.render()
            rewards.append(reward)
            a1 = algo.get_irl_params()
            if done:
                break
   
env.close()

        
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')
n = 50
np.random.seed(10)

obs = [[[0 for k in range(8)] for j in range(n)] for i in range(n)]
acts = [[[0 for k in range(2)] for j in range(n)] for i in range(n)]
energy = [[[0 for k in range(1)] for j in range(n)] for i in range(n)]

with tf.Session():
#        algo.train()
    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    if algo.init_pol_params is not None:
        algo.policy.set_param_values(algo.init_pol_params)
    if algo.init_irl_params is not None:
        algo.irl_model.set_params(algo.init_irl_params)
    algo.start_worker()

#    a = algo.irl_model.eval(paths, gamma=algo.discount, itr=1)
    


    x = np.linspace(2, 24, n)
    y = np.linspace(-11, 11, n)

    X, Y = np.meshgrid(x, y)
    
    for i in range(n):
        for j in range(n):
            obs[i][j][0] = 3.25
            obs[i][j][1] = 0
            obs[i][j][2] = 7.75
            obs[i][j][3] = 0
            obs[i][j][4] = 3.25
            obs[i][j][5] = 0
            obs[i][j][6] = 7.75
            
            obs[i][j][7] = X[i][j]
            acts[i][j][1] = Y[i][j]
            energy[i][j]=    (tf.get_default_session().run(algo.irl_model.energy,
                                                feed_dict={algo.irl_model.act_t: np.array([acts[i][j]]), algo.irl_model.obs_t: np.array([obs[i][j]])}))[0][0]*-1
    

            
            
    plt.contour(X, Y, energy,50, cmap='RdGy')
    plt.colorbar()
    plt.xlabel('long. distance (m)', fontsize=18)
    plt.ylabel('acceleration (m/s/s)', fontsize=16)
      
    
    
    
   
    
    
with tf.Session():
#        algo.train()
    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    if algo.init_pol_params is not None:
        algo.policy.set_param_values(algo.init_pol_params)
    if algo.init_irl_params is not None:
        algo.irl_model.set_params(algo.init_irl_params)
    algo.start_worker()

#    a = algo.irl_model.eval(paths, gamma=algo.discount, itr=1)
    


    x = np.linspace(6.5, 9, n)-6.5
    y = np.linspace(-11, 11, n)

    X, Y = np.meshgrid(x, y)
    
    for i in range(n):
        for j in range(n):
            obs[i][j][0] = 3.25
            obs[i][j][1] = 0
            obs[i][j][3] = 0
            obs[i][j][4] = 3.25
            obs[i][j][5] = 0
            obs[i][j][6] = 6.5
            obs[i][j][7] = 10
            
            obs[i][j][2] = X[i][j]+6.5
            acts[i][j][1] = Y[i][j]
            energy[i][j]=    (tf.get_default_session().run(algo.irl_model.energy,
                                                feed_dict={algo.irl_model.act_t: np.array([acts[i][j]]), algo.irl_model.obs_t: np.array([obs[i][j]])}))[0][0]*-1
    

            
            
    plt.contour(X, Y, energy,50, cmap='RdGy')
    plt.colorbar()
    plt.xlabel('lat. distance (m)', fontsize=18)
    plt.ylabel('acceleration (m/s/s)', fontsize=16)
    
    

with tf.Session():
#        algo.train()
    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    if algo.init_pol_params is not None:
        algo.policy.set_param_values(algo.init_pol_params)
    if algo.init_irl_params is not None:
        algo.irl_model.set_params(algo.init_irl_params)
    algo.start_worker()

#    a = algo.irl_model.eval(paths, gamma=algo.discount, itr=1)
    


    x = np.linspace(0, 6.5, n)
    y = np.linspace(-11, 11, n)

    X, Y = np.meshgrid(x, y)
    
    for i in range(n):
        for j in range(n):
            obs[i][j][1] = 0
            obs[i][j][2] = 7.75

            obs[i][j][3] = 0
            obs[i][j][4] = 3.25
            obs[i][j][5] = 0
            obs[i][j][6] = 6.5
            obs[i][j][7] = 10
            
            obs[i][j][0] = X[i][j]
            acts[i][j][1] = Y[i][j]
            energy[i][j]=    (tf.get_default_session().run(algo.irl_model.energy,
                                                feed_dict={algo.irl_model.act_t: np.array([acts[i][j]]), algo.irl_model.obs_t: np.array([obs[i][j]])}))[0][0]*-1
    

            
            
    plt.contour(X, Y, energy,50, cmap='RdGy')
    plt.colorbar()
    plt.xlabel('speed (m/s)', fontsize=18)
    plt.ylabel('acceleration (m/s/s)', fontsize=16)
        
    
    


with tf.Session():
#        algo.train()
    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    if algo.init_pol_params is not None:
        algo.policy.set_param_values(algo.init_pol_params)
    if algo.init_irl_params is not None:
        algo.irl_model.set_params(algo.init_irl_params)
    algo.start_worker()

#    a = algo.irl_model.eval(paths, gamma=algo.discount, itr=1)
    


    x = np.linspace(2, 24, n)
    y = np.linspace(-80, 80, n)

    X, Y = np.meshgrid(x, y)
    
    for i in range(n):
        for j in range(n):
            obs[i][j][0] = 3.25
            obs[i][j][1] = 0
            obs[i][j][2] = 7.75
            obs[i][j][3] = 0
            obs[i][j][4] = 3.25
            obs[i][j][5] = 0
            obs[i][j][6] = 7.75
            
            obs[i][j][7] = X[i][j]
            acts[i][j][0] = Y[i][j]
            energy[i][j]=    (tf.get_default_session().run(algo.irl_model.energy,
                                                feed_dict={algo.irl_model.act_t: np.array([acts[i][j]]), algo.irl_model.obs_t: np.array([obs[i][j]])}))[0][0]*-1
    

            
            
    plt.contour(X, Y, energy,50, cmap='RdGy')
    plt.colorbar()
    plt.xlabel('long. distance (m)', fontsize=18)
    plt.ylabel('yaw rate', fontsize=16)
      
    
    
    
   
    
    
with tf.Session():
#        algo.train()
    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    if algo.init_pol_params is not None:
        algo.policy.set_param_values(algo.init_pol_params)
    if algo.init_irl_params is not None:
        algo.irl_model.set_params(algo.init_irl_params)
    algo.start_worker()

#    a = algo.irl_model.eval(paths, gamma=algo.discount, itr=1)
    


    x = np.linspace(6.5, 9, n)-6.5
    y = np.linspace(-80, 80, n)

    X, Y = np.meshgrid(x, y)
    
    for i in range(n):
        for j in range(n):
            obs[i][j][0] = 3.25
            obs[i][j][1] = 0
            obs[i][j][3] = 0
            obs[i][j][4] = 3.25
            obs[i][j][5] = 0
            obs[i][j][6] = 6.5
            obs[i][j][7] = 10
            
            obs[i][j][2] = X[i][j]+6.5
            acts[i][j][0] = Y[i][j]
            energy[i][j]=    (tf.get_default_session().run(algo.irl_model.energy,
                                                feed_dict={algo.irl_model.act_t: np.array([acts[i][j]]), algo.irl_model.obs_t: np.array([obs[i][j]])}))[0][0]*-1
    

            
            
    plt.contour(X, Y, energy,50, cmap='RdGy')
    plt.colorbar()
    plt.xlabel('lat. distance (m)', fontsize=18)
    plt.ylabel('yaw rate', fontsize=16)
    
    

with tf.Session():
#        algo.train()
    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    if algo.init_pol_params is not None:
        algo.policy.set_param_values(algo.init_pol_params)
    if algo.init_irl_params is not None:
        algo.irl_model.set_params(algo.init_irl_params)
    algo.start_worker()

#    a = algo.irl_model.eval(paths, gamma=algo.discount, itr=1)
    


    x = np.linspace(0, 6.5, n)
    y = np.linspace(-80, 80, n)

    X, Y = np.meshgrid(x, y)
    
    for i in range(n):
        for j in range(n):
            obs[i][j][1] = 0
            obs[i][j][2] = 7.75

            obs[i][j][3] = 0
            obs[i][j][4] = 3.25
            obs[i][j][5] = 0
            obs[i][j][6] = 6.5
            obs[i][j][7] = 10
            
            obs[i][j][0] = X[i][j]
            acts[i][j][0] = Y[i][j]
            energy[i][j]=    (tf.get_default_session().run(algo.irl_model.energy,
                                                feed_dict={algo.irl_model.act_t: np.array([acts[i][j]]), algo.irl_model.obs_t: np.array([obs[i][j]])}))[0][0]*-1
    

            
            
    plt.contour(X, Y, energy,50, cmap='RdGy')
    plt.colorbar()
    plt.xlabel('speed (m/s)', fontsize=18)
    plt.ylabel('yaw rate', fontsize=16)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import csv

with open('D:/github_clones/AIRL/inverse_rl/data/reward.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(unpacked)


import numpy as np
np.savetxt("D:/github_clones/AIRL/inverse_rl/data/reward.csv", unpacked, delimiter=",", fmt='%s')


