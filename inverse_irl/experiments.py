#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:34:21 2020

@author: hossam
"""

import gym
import gym_bicycle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inverse_irl.csv2pickle import csv2pickle1
from statsmodels.tsa.stattools import acf, pacf
import altair as alt
import altair_saver
import seaborn as sns

env = gym.make('bicycle-v0', num_agents = 2)

simulated_df = pd.read_csv('data/simulated_trajs_gail.csv')
expert_training = csv2pickle1()


observed_df = pd.DataFrame(np.array([[0,0,0,0,0]]),
                   columns=['ID','lat_dist','long_dist', 'speed', 'speed_diff'])


for i in range(len(expert_training)):
    for j in range(len(expert_training[i]['observations'])):
        ID = i
        lat_dist = expert_training[i]['observations'][j][6]-expert_training[i]['observations'][j][2]
        long_dist = expert_training[i]['observations'][j][7]-expert_training[i]['observations'][j][3]
        speed = expert_training[i]['observations'][j][0]
        speed_diff = expert_training[i]['observations'][j][0]-expert_training[i]['observations'][j][4]
        # arr = [ID,long_dist, speed, speed_diff]
        
        observed_df = observed_df.append({'ID': ID,'lat_dist':lat_dist, 'long_dist': long_dist, 'speed': speed, 'speed_diff': speed_diff}, ignore_index=True)
    


env.reset()

jiang_df = pd.read_csv('jiang_df.csv')

zhao_df = pd.read_csv('zhao_df.csv')


jitter = pd.read_csv('jitter.csv')
jitter.columns = ['Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)']

colors=["darkturquoise","mediumpurple","springgreen"]

fig, ax = plt.subplots()
ax = sns.stripplot(data=jitter, palette = colors)
ax.set_ylabel('Mean absolute error in longitudinal distance (m)', fontsize=10)
ax.tick_params(labelsize=10)
fig.savefig('jitter.png', dpi = 300)