# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:16:42 2019

@author: Office
"""
import numpy as np
import operator
import gym
import gym_bicycle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

env = gym.make('bicycle-v0')


env.reset()

# read the starting state from the first point of the trajectory

starting_state = np.array([5.946154682, 0.179752998, 8.305617404, 4.182052845, 6.060055086, 0.632448933,	8.347272139, 3.592877773])

# read training data
data_train = pd.read_csv("data/Traj_df_FBF_interaction2_to_python_complete_training.csv")

#create a zero one row trajectory array

follow_traj_follow= [0]*11


# create a list of the unique objects
Obj_ID_unique = np.unique(data_train['Obj_ID'])


i=0
j=0
first_index= 0

traj_lengths = [0]

# loop over the list of unique IDs to get the longest one

for i in range(len(Obj_ID_unique)):
    is_Obj_ID = data_train['Obj_ID']==Obj_ID_unique[i]
    data_test_Obj_ID = data_train[is_Obj_ID]
    data_test_Obj_ID = data_test_Obj_ID.reset_index()
    traj_lengths.append(data_test_Obj_ID.shape[0])
    
index, value = max(enumerate(traj_lengths), key=operator.itemgetter(1))

i = 0
j = 0
index = 300
# filter the trajectory list to get the longest trajectory array
is_Obj_ID = data_train['Obj_ID']==Obj_ID_unique[index-2]
data_test_Obj_ID = data_train[is_Obj_ID]
data_test_Obj_ID = data_test_Obj_ID.reset_index()

# create the first simulated array
for m in range(1): 
    env.state = data_test_Obj_ID.iloc[0].to_numpy()[5:13]
    Traj = np.array([0])
    for j in range(len(data_test_Obj_ID)):
        
        lead_actions = np.array([data_test_Obj_ID['acc_SGF'][j],data_test_Obj_ID['yaw_rate'][j]])
        
        
        with tf.compat.v1.Session():    
            sess = tf.compat.v1.get_default_session()
            sess.run(tf.compat.v1.global_variables_initializer())
            follow_actions = policy.get_action(observation= env.state)
            
        env.step(action=follow_actions[0])
    #        env.render()
        state_action = np.concatenate([env.state,follow_actions[0]])
        state_action = np.concatenate([Traj,state_action])

        follow_traj_follow = np.vstack([follow_traj_follow, state_action])
        
        print("sample number", m, "trajectory number", i, "frame number", j)

    follow_traj_follow = follow_traj_follow[1:]
    
    
follow_traj_follow2= [0]*11



# create the following simulated arrays
for m in range(1,50):
    env.state = data_test_Obj_ID.iloc[0].to_numpy()[5:13]
    Traj = np.array([m])

    for j in range(len(data_test_Obj_ID)):
        
        lead_actions = np.array([data_test_Obj_ID['acc_SGF'][j],data_test_Obj_ID['yaw_rate'][j]])
        
        
        with tf.compat.v1.Session():    
            sess = tf.compat.v1.get_default_session()
            sess.run(tf.compat.v1.global_variables_initializer())
            follow_actions = policy.get_action(observation= env.state)
            
        env.step(action=follow_actions[0], action_lead=lead_actions)
    #        env.render()
        state_action = np.concatenate([env.state,follow_actions[0]])
        state_action = np.concatenate([Traj,state_action])
        follow_traj_follow2 = np.vstack([follow_traj_follow2, state_action])
        
        print("sample number", m, "trajectory number", i, "frame number", j)

    follow_traj_follow2 = follow_traj_follow2[1:]
    follow_traj_follow3 = np.vstack([follow_traj_follow,follow_traj_follow2])
    
  
np.savetxt("standard_error_401-450.csv", follow_traj_follow3, delimiter=",")

env.close()

