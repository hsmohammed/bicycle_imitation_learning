#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:15:50 2020

@author: hossam
"""
import gym
import gym_bicycle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inverse_irl.csv2pickle import csv2pickle1


env = gym.make('bicycle-v0', num_agents = 3)


expert_training = csv2pickle1()


observed_df = pd.DataFrame(np.array([[0,0,0,0]]),
                   columns=['ID','long_dist', 'speed', 'speed_diff'])

for i in range(len(expert_training)):
    for j in range(len(expert_training[i]['observations'])):
        ID = i
        long_dist = expert_training[i]['observations'][j][7]-expert_training[i]['observations'][j][3]
        speed = expert_training[i]['observations'][j][0]
        speed_diff = expert_training[i]['observations'][j][0]-expert_training[i]['observations'][j][4]
        # arr = [ID,long_dist, speed, speed_diff]
        
        observed_df = observed_df.append({'ID': ID, 'long_dist': long_dist, 'speed': speed, 'speed_diff': speed_diff}, ignore_index=True)
    

fig1, ax1 = plt.subplots()
ax1.set_title('Longitudinal distance (m)')
ax1.boxplot(observed_df['long_dist'], showfliers=False)

fig2, ax2 = plt.subplots()
ax2.set_title('Speed (m/s)')
ax2.boxplot(observed_df['speed'], showfliers=False)

fig3, ax3 = plt.subplots()
ax3.set_title('Speed difference (m/s')
ax3.boxplot(observed_df['speed_diff'], showfliers=False)


env.reset()


env.state[1] = 0
env.state[2] = 7.75
env.state[5] = 0
env.state[6] = 7.75
env.state[8] = 3.2
env.state[9] = 0
env.state[10] = 7.75

env.render()
if env.state[0] > 0:
    p = 0.8
else:
    p = 0.3
    

v_max = 4.2
d_c = 0.9
acceleration = True
deceleration = False
v_a = 1.2

d_od = 6

    
def new_step():
    if acceleration:
        env.state[0] = min(env.state[0] + 1, v_max)
        env.state[4] = min(env.state[4] + 1, v_max)
    d = env.state[7]-env.state[3]
    d_lead = env.state[11]-env.state[7]
    d_lead2 = 3
    # deceleration
    if deceleration:
        if d < d_od:
            env.state[0] = min(env.state[0], d, max(d_lead, d_c))
        else:
            env.state[0] = min(env.state[0], d)
        
        if d_lead < d_od:
            env.state[4] = min(env.state[4], d_lead, max(d_lead2, d_c))
        else:
            env.state[4] = min(env.state[4], d_lead)
    v_vir_fol = max(env.state[0] - 1, 0)
    v_vir_lead = max(env.state[4] - 1, 0)
    v_vir_lead2 = max(env.state[7] - 1, 0)
    env.state[0] = min(env.state[0] + min(v_vir_lead,v_a), v_max)
    env.state[4] = min(env.state[4] + min(v_vir_lead2,v_a), v_max)
    randomization = np.random.binomial(1, p)
    if randomization:
        env.state[0] = np.max(env.state[0]-1, 0)
        env.state[4] = np.max(env.state[4]-1, 0)
    
    env.state[3] = env.state[3] + env.state[0]
    env.state[7] = env.state[7] + env.state[4]
    env.state[11] = env.state[11] + env.state[8]
    
    
    print("follow speed=",
          env.state[0],
          "\nlead (1) speed =",
          env.state[4],
          "\nlead (2) speed=",
          env.state[8],
          "\ny follow=",
          env.state[3],
          "\ny lead (1)=",
          env.state[7],
          "\ny lead (2)=",
          env.state[11])
    env.render()

new_step()

env.close()

















# for i in range(2):
#     # acceleration
#     if acceleration:
#         env.state[0] = min(env.state[0] + 1, v_max)
#         env.state[4] = min(env.state[4] + 1, v_max)
#     d = env.state[7]-env.state[3]
#     d_lead = env.state[11]-env.state[7]
#     d_lead2 = 3
#     # deceleration
#     if deceleration:
#         if d < 1:
#             env.state[0] = min(env.state[0], d, max(d_lead, d_c))
#         else:
#             env.state[0] = min(env.state[0], d)
        
#         if d_lead < 1:
#             env.state[4] = min(env.state[4], d_lead, max(d_lead2, d_c))
#         else:
#             env.state[4] = min(env.state[4], d_lead)
#     v_vir_fol = max(env.state[0] - 1, 0)
#     v_vir_lead = max(env.state[4] - 1, 0)
#     v_vir_lead2 = max(env.state[7] - 1, 0)
#     env.state[0] = min(env.state[0] + min(v_vir_lead,v_a), v_max)
#     env.state[4] = min(env.state[4] + min(v_vir_lead2,v_a), v_max)
#     randomization = np.random.binomial(1, p)
#     if randomization:
#         env.state[0] = np.max(env.state[0]-1, 0)
#         env.state[4] = np.max(env.state[4]-1, 0)
    
#     env.state[3] = env.state[3] + env.state[0]
#     env.state[7] = env.state[7] + env.state[4]
#     env.state[11] = env.state[11] + env.state[8]
    
    
#     print(env.state[3], env.state[7], env.state[11])
#     env.render()
    



