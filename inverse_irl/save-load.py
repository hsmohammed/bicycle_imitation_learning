# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:26:11 2019

@author: Office
"""



import shelve

filename='model_output.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
        
        
my_shelf['AIRLStateAction'] = globals()['AIRLStateAction']
my_shelf['BaseSampler'] = globals()['BaseSampler']
my_shelf['DIST_CATEGORICAL'] = globals()['DIST_CATEGORICAL']
my_shelf['GAIL'] = globals()['GAIL']
my_shelf['GAN_GCL'] = globals()['GAN_GCL']
my_shelf['GaussianMLPPolicy'] = globals()['GaussianMLPPolicy']
my_shelf['IRLTRPO'] = globals()['IRLTRPO']
my_shelf['In'] = globals()['In']
my_shelf['LinearFeatureBaseline'] = globals()['LinearFeatureBaseline']
my_shelf['Npath'] = globals()['Npath']
my_shelf['ProgBarCounter'] = globals()['ProgBarCounter']
my_shelf['SharedGlobal'] = globals()['SharedGlobal']
my_shelf['TrainingIterator'] = globals()['TrainingIterator']
my_shelf['TfEnv'] = globals()['TfEnv']
my_shelf['agent_info'] = globals()['agent_info']
my_shelf['agent_infos'] = globals()['agent_infos']

with tf.Session():
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, "out/out")
    
with tf.Session():
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "out/out")

with tf.Session():    
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    tf.saved_model.save(algo, "out\out")

    
    
my_shelf['TfEnv'] = globals()['TfEnv']
my_shelf['TfEnv'] = globals()['TfEnv']
my_shelf['TfEnv'] = globals()['TfEnv']
my_shelf['TfEnv'] = globals()['TfEnv']





'agent_info',
 'agent_infos',
 'algo',
 'ax',
 'batch_size',
 'categorical_log_pdf',
 'csv2pickle1',

my_shelf.close()






from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  
  
init_op = tf.global_variables_initializer()


saver = tf.train.Saver()


with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
  
  
  
import pickle
file = open("data/bicycle_AIRL3/itr_99.pkl", errors='ignore')

object_file = pickle.load(file)