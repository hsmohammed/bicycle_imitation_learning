#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:54:54 2020

@author: hossam
"""
import gym
import gym_bicycle
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('bicycle-v0', num_agents = 3)


env.reset()


env.state[1] = 0
env.state[2] = 7.75
env.state[5] = 0
env.state[6] = 7.75
env.state[9] = 0
env.state[10] = 7.75



def new_step():
    v_max = 10
    d_nmax = 10
    p_r = 0.3

    
    dn1 = env.state[11] - env.state[3]
    dn2 = env.state[7] - env.state[3]
    
    dn3 = env.state[11]- env.state[7]
    
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
    
    env.state[11] = env.state[11] + env.state[7]
    
    env.render()
    
    
    
new_step()

env.close()
