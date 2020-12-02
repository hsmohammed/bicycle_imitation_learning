# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:12:49 2019

@author: Office
"""
import numpy as np
import tensorflow as tf
import gym
import gym_bicycle
import matplotlib.pyplot as plt
import pandas as pd


env = gym.make('bicycle-v0')


env.reset()
#env.render()

data_test = pd.read_csv("data/Traj_df_FBF_interaction2_to_python_complete_test.csv")
follow_traj= [0]*8

Obj_ID_unique = np.unique(data_test['Obj_ID'])
i=0
j=0
first_index= 0
for i in range(len(Obj_ID_unique)):
    j = 0
    is_Obj_ID = data_test['Obj_ID']==Obj_ID_unique[i]
    data_test_Obj_ID = data_test[is_Obj_ID]
    data_test_Obj_ID = data_test_Obj_ID.reset_index()
    env.state = np.array([data_test_Obj_ID['speed_SGF'][j],
                         data_test_Obj_ID['dir_angle'][j],
                         data_test_Obj_ID['X_axis2_SGF'][j],
                         data_test_Obj_ID['Y_axis2_SGF'][j],
                         data_test_Obj_ID['speed_lead'][j],
                         data_test_Obj_ID['dir_angle_lead'][j],
                         data_test_Obj_ID['X_axis2_SGF_lead'][j],
                         data_test_Obj_ID['Y_axis2_SGF_lead'][j]])
#    env.render()
    for j in range(len(data_test_Obj_ID)):
        
        lead_actions = np.array([data_test_Obj_ID['acc_SGF'][j],data_test_Obj_ID['yaw_rate'][j]])
        
        
        with tf.Session():    
            sess = tf.get_default_session()
            sess.run(tf.global_variables_initializer())
            follow_actions = policy.get_action(observation= env.state)
            
        env.step(action=lead_actions, action_lead=lead_actions)
#        env.render()
        state_action = np.concatenate([env.state,follow_actions[0]])
        follow_traj = np.vstack([follow_traj, env.state])
        
        print("trajectory number", i, "frame number", j)
    

follow_traj = follow_traj[1:]
  
np.savetxt("follow_traj.csv", follow_traj, delimiter=",")

env.close()



















import numpy as np
import tensorflow as tf
import gym
import gym_bicycle
import matplotlib.pyplot as plt
import pandas as pd



env = gym.make('bicycle-v0')


env.reset()



data_train = pd.read_csv("data/Traj_df_FBF_interaction2_to_python_complete_training.csv")
follow_traj_follow= [0]*11

Obj_ID_unique = np.unique(data_train['Obj_ID'])
i=0
j=0
first_index= 0
for i in range(500,550,1):
    j = 0
    is_Obj_ID = data_train['Obj_ID']==Obj_ID_unique[i]
    data_test_Obj_ID = data_train[is_Obj_ID]
    data_test_Obj_ID = data_test_Obj_ID.reset_index()
    env.state = np.array([data_test_Obj_ID['speed_SGF'][j],
                         data_test_Obj_ID['dir_angle'][j],
                         data_test_Obj_ID['X_axis2_SGF'][j],
                         data_test_Obj_ID['Y_axis2_SGF'][j],
                         data_test_Obj_ID['speed_lead'][j],
                         data_test_Obj_ID['dir_angle_lead'][j],
                         data_test_Obj_ID['X_axis2_SGF_lead'][j],
                         data_test_Obj_ID['Y_axis2_SGF_lead'][j]])
#    env.render()
    for j in range(len(data_test_Obj_ID)):
        
        lead_actions = np.array([data_test_Obj_ID['acc_SGF'][j],data_test_Obj_ID['yaw_rate'][j]])
        
        
        with tf.Session():    
            sess = tf.get_default_session()
            sess.run(tf.global_variables_initializer())
            follow_actions = policy.get_action(observation= env.state)
            
        env.step(action=lead_actions, action_lead=lead_actions)
#        env.render()
        state_action = np.concatenate([env.state,follow_actions[0]])
        state_action = np.concatenate([np.array([i]),state_action])
        follow_traj_follow = np.vstack([follow_traj_follow, state_action])
        
        print("trajectory number", i, "frame number", j)
    

    follow_traj_follow = follow_traj_follow[1:]
  
    np.savetxt("follow_traj_follow_500-550.csv", follow_traj_follow, delimiter=",")

env.close()

































