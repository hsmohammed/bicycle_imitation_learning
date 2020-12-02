import numpy as np
import tensorflow as tf
import gym
import gym_bicycle
import matplotlib.pyplot as plt

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
directionAngle_lead_max = np.max(expert_obs[:,5])
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

long_distance_obs = expert_obs[:,3]-expert_obs[:,7]
lat_distane_obs = expert_obs[:,2]-expert_obs[:,6]
speed_obs = expert_obs[:,0]
speed_diff_obs = expert_obs[:,0] - expert_obs[:,4]
cl_dev_obs = expert_obs[:,2] - 7.75
dir_angle_obs = expert_obs[:,1]
dir_angle_diff_obs = expert_obs[:,1] - expert_obs[:,5]

yaw_rate_acts = expert_acts[:,0]
acc_acts = expert_acts[:,1]

yaw_angle_minmax_list = np.linspace(yaw_angle_min, yaw_angle_max, num=num)
acc_minmax_list = np.linspace(acc_min, acc_max, num = num)

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



# yaw rate histogram

ax[0,1].hist(yaw_rate_acts, density=True, bins = 50) 
ax[0,1].set_xlabel("Yaw Rate (Degree/s)")
ax[0,1].xaxis.set_label_coords(0.5, 1.15) 
ax[0,1].set_ylabel("Density")


ax[0,2].axis('off')

# long_distance_hist

ax[1,2].hist(long_distance_obs, density=True, bins = 50, orientation = "horizontal") 
ax[1,2].set_xlabel("Density")
ax[1,2].xaxis.set_label_coords(0.5, 1.15) 

# lat_distance_hist

ax[2,2].hist(lat_distane_obs, density=True, bins = 50, orientation = "horizontal") 
ax[2,2].xaxis.set_label_coords(0.5, 1.15) 


# speed_hist

ax[3,2].hist(speed_obs, density=True, bins = 50, orientation = "horizontal") 
ax[3,2].xaxis.set_label_coords(0.5, 1.15) 

# speed_diff_hist

ax[4,2].hist(speed_diff_obs, density=True, bins = 50, orientation = "horizontal") 
ax[4,2].xaxis.set_label_coords(0.5, 1.15) 


# cl_dev_hist

ax[5,2].hist(cl_dev_obs, density=True, bins = 50, orientation = "horizontal") 
ax[5,2].xaxis.set_label_coords(0.5, 1.15) 

# dir_angle_hist

ax[6,2].hist(dir_angle_obs, density=True, bins = 50, orientation = "horizontal") 
ax[6,2].xaxis.set_label_coords(0.5, 1.15) 



# dir_angle_diff_hist

ax[7,2].hist(dir_angle_diff_obs, density=True, bins = 50, orientation = "horizontal") 
ax[7,2].xaxis.set_label_coords(0.5, 1.15) 





# longitudinal distance vs acceleration


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
p1 = ax[1,0].contourf(xx, yy, zgrid) 

ax[1,0].xaxis.set_label_coords(0.5, 1.15) 
ax[1,0].set_ylabel("Longitudinal Distance (m)")



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

ax[7,0].contourf(xx, yy, zgrid) 
ax[7,0].set_ylabel("Direction angle difference (degrees)")

  



























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

ax[1,1].contourf(xx, yy, zgrid) 
ax[1,1].xaxis.set_label_coords(0.5, 1.15) 



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

ax[7,1].contourf(xx, yy, zgrid) 




cbar_ax = fig.add_axes([2.2, 0.5, 0.05, 0.7])
fig.colorbar(p1, cax=cbar_ax)
fig.subplots_adjust(left=.125, bottom=.1, right=2, top=5, wspace=.2, hspace=.2)
fig.savefig('reward2.png', bbox_inches='tight', dpi = 300)



#env.render()

#with tf.Session():    
#    sess = tf.get_default_session()
#    sess.run(tf.global_variables_initializer())
#    done  = False
#    
#    for i in range(100):
#        state = env.reset()
#        action = algo.policy.get_action(state)[0]
#        
#        state_all = np.vstack((state_all, state))
#        action_all = np.vstack((action_all, action))
#    #    state1 = state
#    #    state2 = env.reset()
#    #    state_all = np.stack((state1, state2), axis = 0)
#    #    action1 = algo.policy.get_action(state1)[0]
#    #    action2 = algo.policy.get_action(state2)[0]
#    #    action_all = np.stack((action1, action2), axis = 0)
#    #
#    #    energy = tf.get_default_session().run(algo.irl_model.energy, feed_dict={algo.irl_model.act_t: action_all, algo.irl_model.obs_t: state_all})
#    
#        while True:
#            
#            
##            energy = tf.get_default_session().run(algo.irl_model.energy, feed_dict={algo.irl_model.act_t: action, algo.irl_model.obs_t: state})
#    
#            state1,reward1,done,_ = env.step(action)
#            action = algo.policy.get_action(state1)[0]
#            
#            state_all = np.vstack((state_all, state1))
#            action_all = np.vstack((action_all, action))
#            
#            #            energy = tf.get_default_session().run(algo.irl_model.energy, feed_dict={algo.irl_model.act_t: action, algo.irl_model.obs_t: state})
##            env.render()
#            if done:
#                break
#        if i > 100:
#            
#            break


with tf.Session():    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    energy = tf.get_default_session().run(algo.irl_model.energy, feed_dict={algo.irl_model.act_t: action_all, algo.irl_model.obs_t: state_all})


x = state_all[0:10000,2]
y = action_all[0:10000,1]
xx, yy = np.meshgrid(x, y)

X_grid = np.c_[ np.ravel(xx), np.ravel(yy) ]

z = energy[0:10000,:]
z = np.repeat(z,10000)
z = z.reshape(xx.shape)

plt.contour(xx, yy, z)  




states = np.array([0,0,0,0,0,0,0,0], dtype=np.float64)
actions = np.array([0,0], dtype=np.float64)

for i in range(999):
    state = np.array([0,0,0,0,0,0,0,0])
    action = np.array([0,0])
    states = np.vstack((states,state))
    actions = np.vstack((actions, action))
    

    
states[:,0] = 3.5
states[:,1] = 0
a = np.linspace(6.5,7.75,100)
b = a
for i in range(9):
    
    b = np.concatenate((a,b))

states[:,2] = b



d = np.repeat(5,100)
for i in range(9):
    c = np.repeat(i+6,100)
    d = np.concatenate((d,c))
    
states[:,3] = d

states[:,4] = 2
states[:,5] = 0
states[:,6] = 7.75
states[:,7] = 11

lat_dist = states[:,2] - states[:,6]
long_dist = states[:,7] - states[:,3]

x = np.linspace(min(lat_dist), max(lat_dist), 100)
y = np.linspace(min(long_dist), max(long_dist), 100)

X, Y = np.meshgrid(x, y)

action_mesh_yaw = np.zeros((100,100))

action_mesh_acc = np.zeros((100,100))

with tf.Session():    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for j in range(100):
            states_point = states[0,:]
            states_point[2] = X[i,j] +7.75
            states_point[3] = 11 - Y[i,j]
            action_mesh_yaw[i,j] = algo.policy.get_action(states_point)[0][0]
            action_mesh_acc[i,j] = algo.policy.get_action(states_point)[0][1]
    energy = tf.get_default_session().run(algo.irl_model.energy, feed_dict={algo.irl_model.act_t: actions, algo.irl_model.obs_t: states})

plt.contour(X, Y, action_mesh_yaw, 1, cmap='RdGy')
plt.contour(X, Y, action_mesh_acc,0.5, cmap='RdGy')

plt.plot(actions[:,1])
plt.plot(energy)


z = griddata()

#        self.min_speed_follow = 0
#        self.min_dirAngle_follow = -55
#        self.min_x_follow = 6.5
#        self.min_y_follow = 0

#        self.max_speed_follow = 6.5
#        self.max_dirAngle_follow = 55
#        self.max_x_follow = 9
#        self.max_y_follow = 50
#
#        
#        self.min_yawRate_follow = -80
#        self.min_Acc_follow = -11
#
#        
#        self.max_yawRate_follow = 80
#        self.max_Acc_follow = 11
    
    
#        self.min_speed_lead = np.zeros(self.num_agents - 1)
#        self.min_dirAngle_lead = np.zeros(self.num_agents - 1)
#        self.min_x_lead = [0] * (self.num_agents - 1)
#        self.min_y_lead = [0] * (self.num_agents - 1)
#        
#        self.max_speed_lead = np.zeros(self.num_agents - 1)
#        self.max_dirAngle_lead = np.zeros(self.num_agents - 1)
#        self.max_x_lead = [0] * (self.num_agents - 1)
#        self.max_y_lead = [0] * (self.num_agents - 1)
#        
#        self.min_yawRate_lead = np.zeros(self.num_agents - 1)
#        self.min_Acc_lead = np.zeros(self.num_agents - 1)
#        
#        self.max_yawRate_lead = np.zeros(self.num_agents - 1)
#        self.max_Acc_lead = np.zeros(self.num_agents - 1)

from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
# make up data.
#npts = int(raw_input('enter # of random points to plot:'))
seed(0)
npts = 200
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = x*np.exp(-x**2 - y**2)
# define grid.
xi = np.linspace(-2.1, 2.1, 100)
yi = np.linspace(-2.1, 2.1, 200)
# grid the data.
zi = griddata(x, y, z, xi, yi, interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = plt.contourf(xi, yi, zi, 15,
                  vmax=abs(zi).max(), vmin=-abs(zi).max())
plt.colorbar()  # draw colorbar
# plot data points.
plt.scatter(x, y, marker='o', s=5, zorder=10)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('griddata test (%d points)' % npts)
plt.show()
