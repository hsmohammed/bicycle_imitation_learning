# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 07:27:54 2019

@author: Office
"""

import tensorflow as tf

saver = tf.train.import_meta_graph('out/out.meta')

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('out/out.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('out/'))
