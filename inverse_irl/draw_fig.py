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
        
        z[i] = energy_long_acc[i]
    
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
        
        z[i] = energy_long_yaw[i]
    
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