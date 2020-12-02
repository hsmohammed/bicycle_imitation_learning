# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 03:11:39 2019

@author: Office
"""
env = gym.make('bicycle-v0')
state_all = np.array([0,0,0,0,0,0,0,0])
action_all = np.array([0,0])
state0 = env.reset()


# variable means

num = 20

speed_follow_mean = np.mean(expert_obs[:,0])
directionAngle_follow_mean = np.mean(expert_obs[:,1])
x_follow_mean = np.mean(expert_obs[:,2])
y_follow_mean = np.mean(expert_obs[:,3])
speed_lead_mean = np.mean(expert_obs[:,4])
directionAngle_lead_mean = np.mean(expert_obs[:,5])
x_lead_mean = np.mean(expert_obs[:,6])
y_lead_mean = np.mean(expert_obs[:,7])


speed_follow_max = np.max(expert_obs[:,0])
directionAngle_follow_max = np.max(expert_obs[:,1])
x_follow_max = np.max(expert_obs[:,2])
y_follow_max = np.max(expert_obs[:,3])
speed_lead_max = np.max(expert_obs[:,4])
directionAngle_lead_max = np.mean(expert_obs[:,5])
x_lead_max = np.max(expert_obs[:,6])
y_lead_max = np.max(expert_obs[:,7])


speed_follow_min = np.min(expert_obs[:,0])
directionAngle_follow_min = np.min(expert_obs[:,1])
x_follow_min = np.min(expert_obs[:,2])
y_follow_min = np.min(expert_obs[:,3])
speed_lead_min = np.min(expert_obs[:,4])
directionAngle_lead_min = np.min(expert_obs[:,5])
x_lead_min = np.min(expert_obs[:,6])
y_lead_min = np.min(expert_obs[:,7])



yaw_angle_mean = np.mean(expert_acts[:,0])
acc_mean = np.mean(expert_acts[:,1])

yaw_angle_max = np.max(expert_acts[:,0])
acc_max = np.max(expert_acts[:,1])

yaw_angle_min = np.min(expert_acts[:,0])
acc_min = np.min(expert_acts[:,1])

yaw_angle_minmax_list = np.linspace(-30, 30, num=num)
acc_minmax_list = np.linspace(-5, 5, num = num)

speed_follow_minmax_list = np.linspace(speed_follow_min, speed_follow_max, num=num)
directionAngle_follow_minmax_list = np.linspace(directionAngle_follow_min, directionAngle_follow_max, num=num)
x_follow_minmax_list = np.linspace(x_follow_min, x_follow_max, num=num)
y_follow_minmax_list = np.linspace(y_follow_min, y_follow_max, num=num)

speed_lead_minmax_list = np.linspace(speed_lead_min, speed_lead_max, num=num)
directionAngle_lead_minmax_list = np.linspace(directionAngle_lead_min, directionAngle_lead_max, num=num)
x_lead_minmax_list = np.linspace(x_lead_min, x_lead_max, num=num)
y_lead_minmax_list = np.linspace(y_lead_min, y_lead_max, num=num)

speed_follow_min_list = np.linspace(speed_follow_min, speed_follow_min, num=num)
directionAngle_follow_min_list = np.linspace(directionAngle_follow_min, directionAngle_follow_min, num=num)
x_follow_min_list = np.linspace(x_follow_min, x_follow_min, num=num)
y_follow_min_list = np.linspace(y_follow_min, y_follow_min, num=num)

speed_lead_min_list = np.linspace(speed_lead_min, speed_lead_min, num=num)
directionAngle_lead_min_list = np.linspace(directionAngle_lead_min, directionAngle_lead_min, num=num)
x_lead_min_list = np.linspace(x_lead_min, x_lead_min, num=num)
y_lead_min_list = np.linspace(y_lead_min, y_lead_min, num=num)

speed_follow_max_list = np.linspace(speed_follow_max, speed_follow_max, num=num)
directionAngle_follow_max_list = np.linspace(directionAngle_follow_max, directionAngle_follow_max, num=num)
x_follow_max_list = np.linspace(x_follow_max, x_follow_max, num=num)
y_follow_max_list = np.linspace(y_follow_max, y_follow_max, num=num)

speed_lead_max_list = np.linspace(speed_lead_max, speed_lead_max, num=num)
directionAngle_lead_max_list = np.linspace(directionAngle_lead_max, directionAngle_lead_max, num=num)
x_lead_max_list = np.linspace(x_lead_max, x_lead_max, num=num)
y_lead_max_list = np.linspace(y_lead_max, y_lead_max, num=num)


speed_follow_mean_list = np.linspace(speed_follow_mean, speed_follow_mean, num=num)
directionAngle_follow_mean_list = np.linspace(directionAngle_follow_mean, directionAngle_follow_mean, num=num)
x_follow_mean_list = np.linspace(x_follow_mean, x_follow_mean, num=num)
y_follow_mean_list = np.linspace(y_follow_mean, y_follow_mean, num=num)

speed_lead_mean_list = np.linspace(speed_lead_mean, speed_lead_mean, num=num)
directionAngle_lead_mean_list = np.linspace(directionAngle_lead_mean, directionAngle_lead_mean, num=num)
x_lead_mean_list = np.linspace(x_lead_mean, x_lead_mean, num=num)
y_lead_mean_list = np.linspace(y_lead_mean, y_lead_mean, num=num)


# long distance





state_100_longDist_array = np.array([speed_follow_mean, 
                                   directionAngle_follow_mean, 
                                   x_follow_mean,
                                   y_follow_min,
                                   speed_lead_mean, 
                                   directionAngle_lead_mean, 
                                   x_lead_mean,
                                   y_lead_minmax_list[0],
                                   ])

for i in range(num-1):
    next_array = np.array([speed_follow_mean, 
                           directionAngle_follow_mean, 
                           x_follow_mean,
                           y_follow_min,
                           speed_lead_mean, 
                           directionAngle_lead_mean, 
                           x_lead_mean,
                           y_lead_minmax_list[i+1],
                           ])
    
    state_100_longDist_array = np.vstack([state_100_longDist_array, next_array])
    
    
state_10000_longDist_array = state_100_longDist_array
for i in range(num-1):
    state_10000_longDist_array = np.vstack([state_10000_longDist_array, state_100_longDist_array])

state_10000_longDist_array.shape

    
# lat distance 
    
state_100_latDist_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_minmax_list[0],
                               y_lead_mean,
                               ])

for i in range(num-1):
    next_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_minmax_list[i+1],
                               y_lead_mean,
                               ])

    
    state_100_latDist_array = np.vstack([state_100_latDist_array, next_array])
    
state_10000_latDist_array = state_100_latDist_array
for i in range(num-1):
    state_10000_latDist_array = np.vstack([state_10000_latDist_array, state_100_latDist_array])

state_10000_latDist_array.shape



# speed
    
state_100_speed_array = np.array([speed_follow_minmax_list[0], 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

for i in range(num-1):
    next_array = np.array([speed_follow_minmax_list[i+1], 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

    
    state_100_speed_array = np.vstack([state_100_speed_array, next_array])
    
state_10000_speed_array = state_100_speed_array
for i in range(num-1):
    state_10000_speed_array = np.vstack([state_10000_speed_array, state_100_speed_array])

state_10000_speed_array.shape


# speed diff
    
state_100_speed_diff_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_minmax_list[0], 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

for i in range(num-1):
    next_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_minmax_list[i+1], 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

    
    state_100_speed_diff_array = np.vstack([state_100_speed_diff_array, next_array])
    
state_10000_speed_diff_array = state_100_speed_diff_array
for i in range(num-1):
    state_10000_speed_diff_array = np.vstack([state_10000_speed_diff_array, state_100_speed_diff_array])

state_10000_speed_diff_array.shape



# speed diff
    
state_100_speed_diff_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_minmax_list[0], 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

for i in range(num-1):
    next_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_minmax_list[i+1], 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])
    
    state_100_speed_diff_array = np.vstack([state_100_speed_diff_array, next_array])
    
state_10000_speed_diff_array = state_100_speed_diff_array
for i in range(num-1):
    state_10000_speed_diff_array = np.vstack([state_10000_speed_diff_array, state_100_speed_diff_array])

state_10000_speed_diff_array.shape



# CL dev
    




state_100_CL_dev_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_minmax_list[0],
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

for i in range(num-1):
    next_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_minmax_list[i+1],
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

    
    state_100_CL_dev_array = np.vstack([state_100_CL_dev_array, next_array])
    
state_10000_CL_dev_array = state_100_CL_dev_array
for i in range(num-1):
    state_10000_CL_dev_array = np.vstack([state_10000_CL_dev_array, state_100_CL_dev_array])

state_10000_CL_dev_array.shape

# direction angle

state_100_dir_array = np.array([speed_follow_mean, 
                               directionAngle_follow_minmax_list[0], 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

for i in range(num-1):
    next_array = np.array([speed_follow_mean, 
                               directionAngle_follow_minmax_list[i+1], 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_mean, 
                               x_lead_mean,
                               y_lead_mean,
                               ])

    
    state_100_dir_array = np.vstack([state_100_dir_array, next_array])
    
state_10000_dir_array = state_100_dir_array
for i in range(num-1):
    state_10000_dir_array = np.vstack([state_10000_dir_array, state_100_dir_array])

state_10000_dir_array.shape


# direction angle difference

state_100_dir_diff_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_minmax_list[0], 
                               x_lead_mean,
                               y_lead_mean,
                               ])

for i in range(num-1):
    next_array = np.array([speed_follow_mean, 
                               directionAngle_follow_mean, 
                               x_follow_mean,
                               y_follow_min,
                               speed_lead_mean, 
                               directionAngle_lead_minmax_list[i+1], 
                               x_lead_mean,
                               y_lead_mean,
                               ])

    
    state_100_dir_diff_array = np.vstack([state_100_dir_diff_array, next_array])
    
state_10000_dir_diff_array = state_100_dir_diff_array
for i in range(num-1):
    state_10000_dir_diff_array = np.vstack([state_10000_dir_diff_array, state_100_dir_diff_array])

state_10000_dir_diff_array.shape


    
    
# acceleration

action_100_acc = np.array([yaw_angle_mean, acc_minmax_list[0]])
for i in range(1):
    acc_current = acc_minmax_list[i]
    for j in range(num-1):
        next_array = np.array([yaw_angle_mean,
                               acc_current])
        action_100_acc = np.vstack([action_100_acc, next_array])
    
action_10000_acc = action_100_acc

action_100_acc = np.array([yaw_angle_mean, acc_minmax_list[0]])

for i in range(num-1):
    acc_current = acc_minmax_list[i+1]
    action_100_acc = np.array([yaw_angle_mean, acc_minmax_list[i+1]])
    for j in range(num-1):
        next_array = np.array([yaw_angle_mean,
                               acc_current])
        action_100_acc = np.vstack([action_100_acc, next_array])
    action_10000_acc = np.vstack([action_10000_acc,action_100_acc])

action_100_acc.shape
action_10000_acc.shape


# yaw angle

action_100_yaw = np.array([yaw_angle_minmax_list[0], acc_mean])
for i in range(1):
    yaw_current = yaw_angle_minmax_list[i]
    for j in range(num-1):
        next_array = np.array([yaw_current,
                               acc_mean])
        action_100_yaw = np.vstack([action_100_yaw, next_array])
    
action_10000_yaw = action_100_yaw

action_100_yaw = np.array([yaw_angle_minmax_list[0], acc_mean])

for i in range(num-1):
    yaw_current = yaw_angle_minmax_list[i+1]
    action_100_yaw = np.array([yaw_angle_minmax_list[i+1], acc_mean])
    for j in range(num-1):
        next_array = np.array([yaw_current,
                               acc_mean])
        action_100_yaw = np.vstack([action_100_yaw, next_array])
    action_10000_yaw = np.vstack([action_10000_yaw,action_100_yaw])

action_100_yaw.shape
action_10000_yaw.shape

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 03:10:24 2019

@author: Office
"""
import numpy as np
import tensorflow as tf
import gym
import gym_bicycle
import matplotlib.pyplot as plt


# longitudinal distance vs acceleration
def draw_fig(filename = 'reward.png'):
    

    
    
    
    
    
    
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    y = state_10000_longDist_array[:, 7]-state_10000_longDist_array[:, 3]
    x = action_10000_acc[:, 1]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_long_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_acc, 
                                                algo.irl_model.obs_t: state_10000_longDist_array})
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
     z[i] = energy_long_acc[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    #plt.contourf(xx, yy, zgrid)   
    #plt.xlabel("Acceleration (m/s/s)")
    #plt.ylabel("Longitudinal Distance (m)")
    #cbar = plt.colorbar()
    #cbar.set_label('reward')
    #plt.show()
    fig, ax = plt.subplots(7,2)
    
    fig.subplots_adjust(left=.125, bottom=.1, right=2, top=5, wspace=.2, hspace=.2)
    fig
    
    p1 = ax[0,0].contourf(xx, yy, zgrid) 
    ax[0,0].set_xlabel("Acceleration (m/s/s)")
    ax[0,0].xaxis.set_label_coords(0.5, 1.15) 
    ax[0,0].set_ylabel("Longitudinal Distance (m)")
    
    
    
    # lateral distance vs acceleration
    
    y = state_10000_latDist_array[:, 6]-state_10000_latDist_array[:, 2]
    x = action_10000_acc[:, 1]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_lat_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_acc, 
                                                algo.irl_model.obs_t: state_10000_latDist_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_lat_acc[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[1,0].contourf(xx, yy, zgrid) 
    ax[1,0].set_ylabel("Lateral Distance (m)")
    
    
    
    
    # speed vs acceleration
    
    y = state_10000_speed_array[:, 0]
    x = action_10000_acc[:, 1]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_speed_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_acc, 
                                                algo.irl_model.obs_t: state_10000_speed_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_speed_acc[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[2,0].contourf(xx, yy, zgrid) 
    ax[2,0].set_ylabel("Speed (m/s)")
    
    
    
    # speed diff vs acceleration
    
    y = state_10000_speed_diff_array[:, 4]-state_10000_speed_diff_array[:, 0]
    x = action_10000_acc[:, 1]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_speed_diff_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_acc, 
                                                algo.irl_model.obs_t: state_10000_speed_diff_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_speed_diff_acc[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[3,0].contourf(xx, yy, zgrid) 
    ax[3,0].set_ylabel("Speed difference (m/s)")
    
    
    
    
    
    # CL dev vs acceleration
    
    y = state_10000_CL_dev_array[:, 2]-7.75
    x = action_10000_acc[:, 1]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_CL_dev_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_acc, 
                                                algo.irl_model.obs_t: state_10000_CL_dev_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_CL_dev_acc[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[4,0].contourf(xx, yy, zgrid) 
    ax[4,0].set_ylabel("Deviation from centerline (m/s)")
    
    
    
    
    # dir vs acceleration
    
    y = state_10000_dir_array[:, 1]
    x = action_10000_acc[:, 1]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_dir_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_acc, 
                                                algo.irl_model.obs_t: state_10000_dir_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_dir_acc[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[5,0].contourf(xx, yy, zgrid) 
    ax[5,0].set_ylabel("Direction angle (degrees)")
    
    
    
    
    # dir diff vs acceleration
    
    y = state_10000_dir_diff_array[:, 5]-state_10000_dir_diff_array[:, 1]
    x = action_10000_acc[:, 1]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_dir_diff_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_acc, 
                                                algo.irl_model.obs_t: state_10000_dir_diff_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_dir_diff_acc[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[6,0].contourf(xx, yy, zgrid) 
    ax[6,0].set_ylabel("Direction angle difference (degrees)")
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # longitudinal distance vs yaw
    
    y = state_10000_longDist_array[:, 7]-state_10000_longDist_array[:, 3]
    x = action_10000_yaw[:, 0]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_long_yaw = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_yaw, 
                                                algo.irl_model.obs_t: state_10000_longDist_array})
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
     z[i] = energy_long_yaw[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[0,1].contourf(xx, yy, zgrid) 
    ax[0,1].set_xlabel("Yaw rate")
    ax[0,1].xaxis.set_label_coords(0.5, 1.15) 
    
    
    
    # lateral distance vs acceleration
    
    y = state_10000_latDist_array[:, 6]-state_10000_latDist_array[:, 2]
    x = action_10000_yaw[:, 0]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_lat_yaw = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_yaw, 
                                                algo.irl_model.obs_t: state_10000_latDist_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_lat_yaw[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[1,1].contourf(xx, yy, zgrid) 
    
    
    
    # speed vs acceleration
    
    y = state_10000_speed_array[:, 0]
    x = action_10000_yaw[:, 0]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_speed_yaw = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_yaw, 
                                                algo.irl_model.obs_t: state_10000_speed_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_speed_yaw[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[2,1].contourf(xx, yy, zgrid) 
    
    
    
    # speed diff vs acceleration
    
    y = state_10000_speed_diff_array[:, 4]-state_10000_speed_diff_array[:, 0]
    x = action_10000_yaw[:, 0]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_speed_diff_yaw = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_yaw, 
                                                algo.irl_model.obs_t: state_10000_speed_diff_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_speed_diff_yaw[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[3,1].contourf(xx, yy, zgrid) 
    
    
    
    # CL dev vs acceleration
    
    y = state_10000_CL_dev_array[:, 2]-7.75
    x = action_10000_yaw[:, 0]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_CL_dev_yaw = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_yaw, 
                                                algo.irl_model.obs_t: state_10000_CL_dev_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_CL_dev_yaw[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[4,1].contourf(xx, yy, zgrid) 
    
    
    
    # dir vs acceleration
    
    y = state_10000_dir_array[:, 1]
    x = action_10000_yaw[:, 0]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_dir_yaw = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_yaw, 
                                                algo.irl_model.obs_t: state_10000_dir_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_dir_yaw[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[5,1].contourf(xx, yy, zgrid) 
    
    
    
    # dir diff vs acceleration
    
    y = state_10000_dir_diff_array[:, 5]-state_10000_dir_diff_array[:, 1]
    x = action_10000_yaw[:, 0]
    xx, yy = np.meshgrid(x, y)
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_dir_diff_yaw = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_10000_yaw, 
                                                algo.irl_model.obs_t: state_10000_dir_diff_array})
    
    
    z = np.zeros(num*num)
    
    for i in range(num*num):
        
        z[i] = energy_dir_diff_yaw[i]
    
    zgrid = griddata((x,y),z, (xx, yy))
    
    ax[6,1].contourf(xx, yy, zgrid) 
    
    
    
    
    cbar_ax = fig.add_axes([2.2, 0.5, 0.05, 0.7])
    fig.colorbar(p1, cax=cbar_ax)
    fig.subplots_adjust(left=.125, bottom=.1, right=2, top=5, wspace=.2, hspace=.2)
    fig.savefig(filename, bbox_inches='tight', dpi = 300)


draw_fig(filename = 'reward1')
draw_fig('reward2')
draw_fig('reward3')
draw_fig('reward4')
draw_fig('reward5')
draw_fig('reward6')
draw_fig('reward7')
draw_fig('reward8')

draw_fig('reward9')
draw_fig('reward10')
draw_fig('reward11')
draw_fig('reward12')
draw_fig('reward13')
draw_fig('reward14')
draw_fig('reward15')
draw_fig('reward16')
draw_fig('reward17')
draw_fig('reward18')
draw_fig('reward19')

draw_fig('reward19')
draw_fig('reward20')
draw_fig('reward21')
draw_fig('reward22')
draw_fig('reward23')
draw_fig('reward24')
draw_fig('reward25')
draw_fig('reward26')
draw_fig('reward27')
draw_fig('reward28')


num = 400

def draw_fig_uni(variable = "long_dist", filename = 'reward101.png'):
    
    
    
    action_mean = np.array([yaw_angle_mean, acc_mean])
    
    for i in range(num-1):
        next_array = np.array([yaw_angle_mean,
                               acc_mean])
        action_mean = np.vstack([action_mean, next_array])

    
    
    fig, ax = plt.subplots(6,2)
    fig.subplots_adjust(left=.125, bottom=.1, right=2, top=5, wspace=.2, hspace=.2)

    
    
    
    y = state_10000_longDist_array[:, 7]-state_10000_longDist_array[:, 3]
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_long = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_mean, 
                                                algo.irl_model.obs_t: state_10000_longDist_array})

    ax[0,0].plot(np.sort(y), np.reshape(energy_long[np.argsort(y)],(num,)),'k')
    ax[0,0].set(xlabel='Longitudinal distance (m)', ylabel='Reward')

    ax[0,0].set_yticklabels(['2.2', '2.6', '3.0', '3.4', '3.8', '4.2', '4.6', '5.0', '5.4'])
    ax[0,1].hist(expert_obs[:,7]-expert_obs[:,3], 20, density=True)
    ax[0,1].set(xlabel='Longitudinal distance (m)', ylabel='Density')
    
    
    
    
    y = state_10000_latDist_array[:, 6]-state_10000_latDist_array[:, 2]
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_lat = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_mean, 
                                                algo.irl_model.obs_t: state_10000_latDist_array})

    ax[1,0].plot(np.sort(y), np.reshape(energy_lat[np.argsort(y)],(num,)),'k')
    ax[1,0].set(xlabel='Lateral distance (m)', ylabel='Reward')
    ax[1,0].set_yticklabels(['2.2', '2.6', '3.0', '3.4', '3.8', '4.2', '4.6', '5.0', '5.4'])
    ax[1,1].hist(expert_obs[:,6]-expert_obs[:,2], 20, density=True)
    ax[1,1].set(xlabel='Lateral distance (m)', ylabel='Density')
    
    
    
    y = state_10000_speed_array[:, 0]
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_speed = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_mean, 
                                                algo.irl_model.obs_t: state_10000_latDist_array})

    ax[2,0].plot(np.sort(y), np.reshape(energy_speed[np.argsort(y)],(num,)),'k')
    ax[2,0].set(xlabel='Speed (m/s)', ylabel='Reward')
    ax[2,0].set_yticklabels(['2.2', '2.6', '3.0', '3.4', '3.8', '4.2', '4.6', '5.0', '5.4'])
    ax[2,1].hist(expert_obs[:,0], 20, density=True)
    ax[2,1].set(xlabel='Speed (m/s)', ylabel='Density')
    
    
    
    
    y = state_10000_speed_diff_array[:, 4]-state_10000_speed_diff_array[:, 0]
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_speed_diff = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_mean, 
                                                algo.irl_model.obs_t: state_10000_speed_diff_array})

    ax[3,0].plot(np.sort(y), np.reshape(energy_speed_diff[np.argsort(y)],(num,)),'k')
    ax[3,0].set(xlabel='Speed difference (m/s)', ylabel='Reward')
    ax[3,0].set_yticklabels(['2.2', '2.6', '3.0', '3.4', '3.8', '4.2', '4.6', '5.0', '5.4'])
    ax[3,1].hist(expert_obs[:,0]-expert_obs[:,4], 20, density=True)
    ax[3,1].set(xlabel='Speed difference (m/s)', ylabel='Density')
    
    
    
    
    y = state_10000_dir_array[:, 1]
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_cl_dev = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_mean, 
                                                algo.irl_model.obs_t: state_10000_dir_array})

    ax[4,0].plot(np.sort(y), np.reshape(energy_cl_dev[np.argsort(y)],(num,)),'k')
    ax[4,0].set(xlabel='Direction angle (degree)', ylabel='Reward')
    ax[4,0].set_yticklabels(['2.2', '2.6', '3.0', '3.4', '3.8', '4.2', '4.6', '5.0', '5.4'])
    ax[4,1].hist(expert_obs[:,1], 50, density=True)
    ax[4,1].set(xlabel='Direction angle (degree)', ylabel='Density')
    
    
    
    
    
    y = state_10000_CL_dev_array[:, 2]-7.75
    
    with tf.Session():    
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        energy_long_acc = tf.get_default_session().run(algo.irl_model.energy, 
                                                feed_dict={algo.irl_model.act_t: action_mean, 
                                                algo.irl_model.obs_t: state_10000_CL_dev_array})

    ax[5,0].plot(np.sort(y), np.reshape(energy_long_acc[np.argsort(y)],(num,)),'k')
    ax[5,0].set(xlabel='Deviation from centerline (m)', ylabel='Reward')
    ax[5,0].set_yticklabels(['2.2', '2.6', '3.0', '3.4', '3.8', '4.2', '4.6', '5.0', '5.4'])
    ax[5,1].hist(expert_obs[:,2]-7.75, 20, density=True)
    ax[5,1].set(xlabel='Deviation from centerline (m)', ylabel='Density')
    
    
    fig.savefig(filename, bbox_inches='tight', dpi = 200)
    plt.show()
    
    return fig
    
    
draw_fig_uni(filename = "reward_all125.png") 



draw_fig_uni("lat_dist", "reward102.jpg") 
draw_fig_uni("speed", "reward103.jpg") 
draw_fig_uni("speed_diff", "reward104.jpg") 
draw_fig_uni("dir", "reward105.jpg") 
draw_fig_uni("cl_dev", "reward106.jpg") 
    
    