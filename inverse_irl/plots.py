    # -*- coding: utf-8 -*-
    """
    Created on Mon Dec 16 04:48:30 2019
    
    @author: Office
    """
    
    
    
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    # acc histogram
    fig, ax = plt.subplots(8,3)
    
    fig.subplots_adjust(left=.125, bottom=.1, right=2, top=5, wspace=.2, hspace=.2)
    fig
    
    ax[0,0].hist(acc_acts, density=True, bins = 50) 
    ax[0,0].set_xlabel("Acceleration (m/s/s)")
    ax[0,0].xaxis.set_label_coords(0.5, 1.15) 
    ax[0,0].set_ylabel("Density")
    ax[0,0].set_xlim(-10,10)
    
    
    # yaw rate histogram
    
    ax[0,1].hist(yaw_rate_acts, density=True, bins = 50) 
    ax[0,1].set_xlabel("Yaw Rate (Degree/s)")
    ax[0,1].xaxis.set_label_coords(0.5, 1.15) 
    ax[0,1].set_ylabel("Density")
    ax[0,1].set_xlim(-75,75)

    
    ax[0,2].axis('off')
    
    # long_distance_hist
    
    ax[1,2].hist(long_distance_obs, density=True, bins = 50, orientation = "horizontal") 
    ax[1,2].set_xlabel("Density")
    ax[1,2].xaxis.set_label_coords(0.5, 1.15) 
    ax[1,2].set_ylim(-25,-1)
    
    # lat_distance_hist
    
    ax[2,2].hist(lat_distane_obs, density=True, bins = 50, orientation = "horizontal") 
    ax[2,2].xaxis.set_label_coords(0.5, 1.15) 
    ax[2,2].set_ylim(-1.2,1.1)
    
    
    # speed_hist
    
    ax[3,2].hist(speed_obs, density=True, bins = 50, orientation = "horizontal") 
    ax[3,2].xaxis.set_label_coords(0.5, 1.15) 
    ax[3,2].set_ylim(0,6)
    
    # speed_diff_hist
    
    ax[4,2].hist(speed_diff_obs, density=True, bins = 50, orientation = "horizontal") 
    ax[4,2].xaxis.set_label_coords(0.5, 1.15) 
    ax[4,2].set_ylim(-2,2)
    
    
    # cl_dev_hist
    
    ax[5,2].hist(cl_dev_obs, density=True, bins = 50, orientation = "horizontal") 
    ax[5,2].xaxis.set_label_coords(0.5, 1.15) 
    ax[5,2].set_ylim(-1.2,1.2)
    
    # dir_angle_hist
    
    ax[6,2].hist(dir_angle_obs, density=True, bins = 50, orientation = "horizontal") 
    ax[6,2].xaxis.set_label_coords(0.5, 1.15) 
    ax[6,2].set_ylim(-40,40)
    
    
    
    # dir_angle_diff_hist
    
    ax[7,2].hist(dir_angle_diff_obs, density=True, bins = 50, orientation = "horizontal") 
    ax[7,2].xaxis.set_label_coords(0.5, 1.15)
    ax[7,2].set_ylim(-55,55)

    
    
    
    
    
    # longitudinal distance vs acceleration
    
    
    y = state_10000_longDist_array[:, 3]-state_10000_longDist_array[:, 7]
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
    p1 = ax[1,0].contourf(xx, yy, zgrid) 
    
    ax[1,0].xaxis.set_label_coords(0.5, 1.15) 
    ax[1,0].set_ylabel("Longitudinal Distance (m)")
    ax[1,0].set_xlim(-10,10)
    ax[1,0].set_ylim(-25,-1)
    
    
    
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
    
    ax[2,0].contourf(xx, yy, zgrid) 
    ax[2,0].set_ylabel("Lateral Distance (m)")
    ax[2,0].set_xlim(-10,10)
    ax[2,0].set_ylim(-1.2,1.1)
    
    
    
    
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
    
    ax[3,0].contourf(xx, yy, zgrid) 
    ax[3,0].set_ylabel("Speed (m/s)")
    ax[3,0].set_xlim(-10,10)
    ax[3,0].set_ylim(0,6)
    
    
    
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
    
    ax[4,0].contourf(xx, yy, zgrid) 
    ax[4,0].set_ylabel("Speed difference (m/s)")
    ax[4,0].set_xlim(-10,10)
    ax[4,0].set_ylim(-2,2)
    
    
    
    
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
    
    ax[5,0].contourf(xx, yy, zgrid) 
    ax[5,0].set_ylabel("Deviation from centerline (m/s)")
    ax[5,0].set_xlim(-10,10)
    ax[5,0].set_ylim(-1.2,1.1)
    
    
    
    
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
    
    ax[6,0].contourf(xx, yy, zgrid) 
    ax[6,0].set_ylabel("Direction angle (degrees)")
    ax[6,0].set_xlim(-10,10)
    ax[6,0].set_ylim(-40,40)
    
    
    
    
    # dir diff vs acceleration
    
    y = state_10000_dir_diff_array[:, 1]-state_10000_dir_diff_array[:, 5]
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
    
    ax[7,0].contourf(xx, yy, zgrid) 
    ax[7,0].set_ylabel("Direction angle difference (degrees)")
    ax[7,0].set_xlim(-10,10)
    ax[7,0].set_ylim(-55,55)
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # longitudinal distance vs yaw
    
    y = state_10000_longDist_array[:, 3]-state_10000_longDist_array[:, 7]
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
    
    ax[1,1].contourf(xx, yy, zgrid) 
    ax[1,1].xaxis.set_label_coords(0.5, 1.15) 
    ax[1,1].set_xlim(-75,75)
    ax[1,1].set_ylim(-25,-1)
    
    
    
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
    
    ax[2,1].contourf(xx, yy, zgrid) 
    ax[2,1].set_xlim(-75,75)
    ax[2,1].set_ylim(-1.2,1.1)
    
    
    
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
    
    ax[3,1].contourf(xx, yy, zgrid) 
    ax[3,1].set_xlim(-75,75)
    ax[3,1].set_ylim(0,6)
    
    
    
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
    
    ax[4,1].contourf(xx, yy, zgrid) 
    ax[4,1].set_xlim(-75,75)
    ax[4,1].set_ylim(-2,2)
    
    
    
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
    
    ax[5,1].contourf(xx, yy, zgrid) 
    ax[5,1].set_xlim(-75,75)
    ax[5,1].set_ylim(-1.2,1.1)
    
    
    
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
    
    ax[6,1].contourf(xx, yy, zgrid) 
    ax[6,1].set_xlim(-75,75)
    ax[6,1].set_ylim(-40,40)
    
    
    
    # dir diff vs acceleration
    
    y = state_10000_dir_diff_array[:, 1]-state_10000_dir_diff_array[:, 5]
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
    
    ax[7,1].contourf(xx, yy, zgrid) 
    ax[7,1].set_xlim(-75,75)
    ax[7,1].set_ylim(-55,55)
    
    
    
    
    cbar_ax = fig.add_axes([2.2, 0.5, 0.05, 0.7])
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.set_label('Reward')
    fig.subplots_adjust(left=.125, bottom=.1, right=2, top=5, wspace=.2, hspace=.2)
    fig.savefig('reward6.png', bbox_inches='tight', dpi = 300)
    
