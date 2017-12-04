# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:58:29 2017

@author: Zhang
"""

import tensorflow as tf
import numpy as np
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3
Weight = tf.Variable(tf.random_uniform[1])
biases = tf.Variable(tf.zeros[1])
y = Weight*x_data+biases
loss = tf.reduce_mean(tf.aquare(y-y_data))
optimizer = tf.train.GrandientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weight),sess.run(biases))