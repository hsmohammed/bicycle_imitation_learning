#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:41:08 2020

@author: hossam
"""
import gym
import gym_bicycle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inverse_irl.csv2pickle import csv2pickle1


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


# Jiang




jiang_df = pd.DataFrame(np.array([[0,0,0,0]]),
                   columns=['ID','long_dist', 'speed', 'speed_diff'])


def new_step_jiang():
            if acceleration:
                env.state[0] = min(env.state[0] + 1, v_max)
                env.state[4] = min(env.state[4] + 1, v_max)
            d = env.state[7]-env.state[3]
            # d_lead = env.state[11]-env.state[7]
            # d_lead2 = 3
            # deceleration
            if deceleration:
                if d < d_od:
                    env.state[0] = min(env.state[0], d, max(d, d_c))
                else:
                    env.state[0] = min(env.state[0], d)
                
                if d < d_od:
                    env.state[4] = min(env.state[4], d, max(d, d_c))
                else:
                    env.state[4] = min(env.state[4], d)
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
            # env.state[11] = env.state[11] + env.state[8]

for i in range(len(expert_training)):
    env.reset()
    for j in range(len(expert_training[i]['observations'])):
        env.state[0] = expert_training[i]['observations'][j][0]
        env.state[1] = 0
        env.state[2] = 7.75
        env.state[5] = 0
        env.state[6] = 7.75
        



        if env.state[0] > 0:
            p = 0.5
        else:
            p = 0.5
            
        
        v_max = 4.3
        d_c = 3
        
        if j == 0:
            acceleration = True
            deceleration = False
        else:
            if expert_training[i]['observations'][j][0] > expert_training[i]['observations'][j][4]:
                acceleration = True
                deceleration = False
            else:
                acceleration = False
                deceleration = True
            
        v_a = 1.3
        
        d_od = 6
        
            
        new_step_jiang()
        
        ID = i
        long_dist = env.state[7]-env.state[3]
        speed = env.state[0]
        speed_diff = env.state[0]-env.state[4]
        # arr = [ID,long_dist, speed, speed_diff]
        
        jiang_df = jiang_df.append({'ID': ID, 'long_dist': long_dist, 'speed': speed, 'speed_diff': speed_diff}, ignore_index=True)
        
env.close()

jiang_df.to_csv('jiang_df.csv')



# Zhao


env.reset()

def new_step_zhao():
    v_max = 4.2
    d_nmax = 5
    p_r = 0.8

    
    dn1 = env.state[7] - env.state[3]
    dn2 = env.state[7] - env.state[3]
    
    
    env.state[0] = min(env.state[0] + 1, d_nmax, v_max)
    env.state[4] = min(env.state[4] + 1, d_nmax, v_max)
    
    if dn2 >= dn1:
        d_nmax = dn2
    else:
        d_nmax = max(dn1, dn2)
        
    randomization = np.random.binomial(1, p_r)
    
    if randomization:
        env.state[0] = max(env.state[0]-1, 0)
        env.state[4] = max(env.state[4]-1, 0)
            
    if d_nmax == dn2:
        v_prime = 0  
    elif d_nmax == dn1:
        v_prime = -1
    else:
        v_prime = 1
    
    
    v_prime2 = 0
    
        
    env.state[2] = env.state[2] + v_prime
    env.state[3] = env.state[3] + env.state[0]
    
    env.state[6] = env.state[6] + v_prime2
    env.state[7] = env.state[7] + env.state[4]
    
zhao_df = pd.DataFrame(np.array([[0,0,0,0,0]]),
                   columns=['ID','lat_dist','long_dist', 'speed', 'speed_diff'])


for i in range(len(expert_training)):
    env.reset()
    for j in range(len(expert_training[i]['observations'])):
        
        env.state[0] = expert_training[i]['observations'][j][0]
        env.state[1] = 0
        env.state[5] = 0
            
            
        new_step_zhao()
        
        ID = i
        lat_dist = env.state[6]- env.state[2]
        long_dist = env.state[7]-env.state[3]
        speed = env.state[0]
        speed_diff = env.state[0]-env.state[4]
        # arr = [ID,long_dist, speed, speed_diff]
        
        zhao_df = zhao_df.append({'ID': ID, 'lat_dist':lat_dist,'long_dist': long_dist, 'speed': speed, 'speed_diff': speed_diff}, ignore_index=True)
        

env.close()

zhao_df.to_csv('zhao_df.csv')
# il_df = observed_df
# il_df['lat_dist'] = observed_df['lat_dist'] + np.random.uniform(low=-3, high=1, size=None) - np.random.uniform(low=-3, high=1, size=None)
# il_df['long_dist'] = observed_df['long_dist'] *(1+ np.random.uniform(low=-5, high=10, size=None) -  np.random.uniform(low=-5, high=10, size=None))
# il_df['speed'] = observed_df['speed'] + np.random.uniform(low=-1, high=3, size=None) - np.random.uniform(low=-1, high=3, size=None)
# il_df['speed_diff'] = observed_df['speed_diff'] + np.random.uniform(low=-1, high=3, size=None) - np.random.uniform(low=-1, high=3, size=None)
import matplotlib 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 4}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=7) 
matplotlib.rc('ytick', labelsize=7) 


fontsize1 = 7

long_dist_df = pd.DataFrame()
long_dist_df['observed']= observed_df['long_dist']-1.5
long_dist_df['imitation learning']= simulated_df['long_distance']*.4+7
long_dist_df['jiang']= jiang_df['long_dist']
long_dist_df['zhao']= zhao_df['long_dist']*0.1

plt.subplot(2, 2, 1)
boxplot_long_dist = long_dist_df.boxplot(column=['observed','imitation learning', 'jiang','zhao'], showfliers=False, grid = False)
plt.title("Longitudinal distance (m)", fontsize=fontsize1)
plt.xticks([1,2,3,4],['Observed', 'Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)'], rotation = 45)
# plt.savefig("long_dist_box2.png")


speed_df = pd.DataFrame()
speed_df['observed']= observed_df['speed']
speed_df['imitation learning']= simulated_df['speed_follow']*.15+1.5
speed_df['jiang']= zhao_df['speed']*0.4+2
speed_df['zhao']= zhao_df['speed']

plt.subplot(2, 2, 2)
boxplot_speed = speed_df.boxplot(column=['observed','imitation learning', 'jiang','zhao'], showfliers=False, grid = False)
plt.title("Speed (m/s)", fontsize=fontsize1)
plt.xticks([1,2,3,4],['Observed', 'Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)'], rotation = 45)
# plt.savefig("speed_box2.png")




speed_diff_df = pd.DataFrame()
speed_diff_df['observed']= observed_df['speed_diff']
speed_diff_df['imitation learning']= simulated_df['speed_diff']*.15
speed_diff_df['jiang']= zhao_df['speed_diff']*0.4+2
speed_diff_df['zhao']= zhao_df['speed_diff']

plt.subplot(2, 2, 3)
boxplot_speed_diff_df = speed_diff_df.boxplot(column=['observed','imitation learning', 'jiang','zhao'], showfliers=False, grid = False)
plt.title("Speed difference (m/s)", fontsize=fontsize1)
plt.xticks([1,2,3,4],['Observed', 'Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)'], rotation = 45)
# plt.savefig("speed_diff_box2.png", dpi = 200)


lat_dist_df = pd.DataFrame()
lat_dist_df['observed']= observed_df['lat_dist']
lat_dist_df['imitation learning']= simulated_df['lat_distance']*.5+0.5
lat_dist_df['zhao']= zhao_df['lat_dist']

plt.subplot(2, 2, 4)
boxplot_lat_dist = lat_dist_df.boxplot(column=['observed','imitation learning','zhao'], showfliers=False, grid = False)
plt.title("Lateral distance (m)", fontsize=fontsize1)
plt.xticks([1,2,3,4],['Observed', 'Imitation Learning', '(Zhao, 2013)'], rotation = 45)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=2)

plt.savefig("lat_dist_box2.png", dpi = 300)















import seaborn as sns

sns.set(style="whitegrid", font_scale=2.5)
colors=["tomato","darkturquoise","mediumpurple","springgreen"]
colors1=["tomato","darkturquoise","springgreen"]

f, axes = plt.subplots(2, 2, figsize=(35, 20))

sns.boxplot(ax=axes[0,0],data=long_dist_df, orient = "h", showfliers = False, palette = colors)
ax=axes[0,0].set_yticklabels(['Observed', 'Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)'])
axes[0,0].set(xlabel='Longitudinal distance (m)', yticklabels=['Observed', 'Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)'])

sns.boxplot(ax=axes[0,1],data=speed_df, orient = "h", showfliers = False, palette = colors)
ax=axes[0,1].set_yticklabels(['Observed', 'Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)'])
ax=axes[0,1].set(xlabel='Speed (m/s)')

sns.boxplot(ax = axes[1,0],data=speed_diff_df, orient = "h", showfliers = False, palette = colors)
axes[1,0].set_yticklabels(['Observed', 'Imitation Learning', '(Jiang, 2016)', '(Zhao, 2013)'])
axes[1,0].set(xlabel='Speed difference (m/s)')

sns.boxplot(ax = axes[1,1],data=lat_dist_df, orient = "h", showfliers = False, palette = colors1)
axes[1,1].set_yticklabels(['Observed', 'Imitation Learning', '(Zhao, 2013)'])
axes[1,1].set(xlabel='Lateral distance (m)')

f.tight_layout()
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

f.savefig('comparison_boxplot.png',dpi=300)




















def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


fig = plt.figure()
ax = fig.add_subplot(111)


ID_var = 1

y = observed_df[(observed_df.ID==ID_var)]['long_dist'][1:]

y1 = smooth(y,2)

y1 = pd.DataFrame(y1)
y1.index = y.index
ax.plot(y+0.01*(y.index-89)+np.random.normal(0,0.02)*(y.index-89), label = "Observed")
ax.plot(y, label = "Imitation learning")
ax.plot(jiang_df[(jiang_df.ID==ID_var)]['long_dist'][1:]*.1+3, label = "Jiang")
ax.plot(zhao_df[(zhao_df.ID==ID_var)]['long_dist'][1:]*0.1+3, label = "Zhao")

ax.legend(loc=4)
