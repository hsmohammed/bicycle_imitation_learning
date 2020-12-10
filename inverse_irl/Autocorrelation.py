#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:33:39 2020

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


def acf1(x):
    y = acf(x)
    y_mean = np.mean(y)
    
    return y_mean

observed_acf = observed_df.groupby('ID', as_index=False).agg({'lat_dist': acf1, 'long_dist' : acf1, 'speed' : acf1, 'speed_diff' : acf1})

simulated_acf = simulated_df.groupby('Traj', as_index=False).agg({'lat_distance': acf1, 'long_distance' : acf1, 'speed_follow' : acf1, 'speed_diff' : acf1})

simulated_acf.columns = ['ID', 'lat_dist', 'long_dist', 'speed', 'speed_diff']

jiang_acf = jiang_df.groupby('ID', as_index=False).agg({'long_dist' : acf1, 'speed' : acf1, 'speed_diff' : acf1})

zhao_acf = zhao_df.groupby('ID', as_index=False).agg({'lat_dist': acf1, 'long_dist' : acf1, 'speed' : acf1, 'speed_diff' : acf1})

merged = simulated_acf.merge(observed_acf, left_on='ID', right_on='ID')

merged = merged.merge(jiang_acf, left_on='ID', right_on='ID')

merged = merged.merge(zhao_acf, left_on='ID', right_on='ID')

merged.columns = ['ID', 'lat_dist_observed', 'long_dist_observed', 'speed_observed', 'speed_diff_observed','lat_dist_simulated', 'long_dist_simulated', 'speed_simulated', 'speed_diff_simulated', 'long_dist_jiang', 'speed_jiang', 'speed_diff_jiang', 'lat_dist_zhao', 'long_dist_zhao', 'speed_zhao', 'speed_diff_zhao']

density_long_dist = alt.Chart(merged).transform_fold(
    ['long_dist_observed',
     'long_dist_simulated',
     'long_dist_jiang',
     'long_dist_zhao'],
    as_ = ['Measurement_type', 'value']
).transform_density(
    density='value',
    bandwidth=0.3,
    groupby=['Measurement_type'],
    extent= [0, 8],
    counts = True,
    steps=200
).mark_area().encode(
    alt.X('value:Q'),
    alt.Y('density:Q', stack='zero'),
    alt.Color('Measurement_type:N')
).properties(width=400, height=100)

density_long_dist.save('density_long_dist.html')






density_lat_dist = alt.Chart(merged).transform_fold(
    ['lat_dist_observed',
     'lat_dist_simulated',
     'lat_dist_zhao'],
    as_ = ['Measurement_type', 'value']
).transform_density(
    density='value',
    bandwidth=0.3,
    groupby=['Measurement_type'],
    extent= [0, 8],
    counts = True,
    steps=200
).mark_area().encode(
    alt.X('value:Q'),
    alt.Y('density:Q', stack='zero'),
    alt.Color('Measurement_type:N')
).properties(width=400, height=100)

density_lat_dist.save('density_lat_dist.html')





density_speed = alt.Chart(merged).transform_fold(
    ['speed_observed',
     'speed_simulated',
     'speed_jiang',
     'speed_zhao'],
    as_ = ['Measurement_type', 'value']
).transform_density(
    density='value',
    bandwidth=0.3,
    groupby=['Measurement_type'],
    extent= [0, 8],
    counts = True,
    steps=200
).mark_area().encode(
    alt.X('value:Q'),
    alt.Y('density:Q', stack='zero'),
    alt.Color('Measurement_type:N')
).properties(width=400, height=100)

density_speed.save('density_speed.html')

plt.density(merged['speed_observed'])

plt.hist(merged['speed_simulated'], bins = 20)




fig, ax = plt.subplots()
for a in [merged['long_dist_observed'], merged['long_dist_simulated'], merged['long_dist_jiang'], merged['long_dist_zhao']]:
    sns.kdeplot(a)
fig.savefig('long_dist_acf.png', dpi = 300)

fig, ax = plt.subplots()
for a in [merged['lat_dist_observed'], merged['lat_dist_simulated'], merged['lat_dist_zhao']]:
    sns.kdeplot(a,  ax=ax)
fig.savefig('lat_dist_acf.png', dpi = 300)

fig, ax = plt.subplots()
for a in [merged['speed_observed'], merged['speed_simulated'], merged['speed_jiang'], merged['speed_zhao']]:
    sns.kdeplot(a,  ax=ax)
fig.savefig('speed_acf.png', dpi = 300)

fig, ax = plt.subplots()
for a in [merged['speed_diff_observed'], merged['speed_diff_simulated'], merged['speed_diff_jiang'], merged['speed_diff_zhao']]:
    sns.kdeplot(a,  ax=ax)
fig.savefig('speed_diff_acf.png', dpi = 300)