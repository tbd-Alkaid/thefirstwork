from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import time
import numpy as np
import tensorflow as tf
import copy
from tensorflow.python.training import moving_averages

TF_DTYPE = tf.float32
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0

class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, sess):
        self._config = config
        self._bsde = bsde
        self._sess = sess
        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self._extra_train_ops = []

    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        dw_valid, x_valid,t_valid = self._bsde.sample(self._config.valid_size,self._num_time_interval,self._bsde.delta_t)
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._t : t_valid,self._is_training: True}
        # initialization
        self._sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        for step in range(self._config.num_iterations+1):
            if step % self._config.logging_frequency == 0:
                loss = self._sess.run(self._newloss, feed_dict=feed_dict_valid)
                # print(loss.shape)
                #########################################
                # y = self._sess.run(self.y, feed_dict=feed_dict_valid)####
                elapsed_time = time.time()-start_time+self._t_build
                training_history.append((loss,elapsed_time))
                if self._config.verbose:
                    logging.info("loss: %.4e,   elapsed time %3u" % (
                        loss,elapsed_time))
            dw_train, x_train,t_train = self._bsde.sample(self._config.valid_size,self._num_time_interval,self._bsde.delta_t)
            self._sess.run(self._train_ops, feed_dict={self._dw: dw_train,
                                                       self._x: x_train,
                                                       self._t: t_train,
                                                       self._is_training: True})
        return np.array(training_history)

    def build(self):
        start_time = time.time()
        self._t = tf.placeholder(TF_DTYPE,self._num_time_interval, name='t')
        self._dw = tf.placeholder(TF_DTYPE, [self._config.valid_size, self._dim, self._num_time_interval], name='dW')
        self._x = tf.placeholder(TF_DTYPE, [self._config.valid_size, self._dim, self._num_time_interval + 1], name='X')
        self._is_training = tf.placeholder(tf.bool,name= 'is_train')
        # self._y_init = tf.Variable(tf.random_uniform([1],
        #                                              minval=self._config.y_init_range[0],
        #                                              maxval=self._config.y_init_range[1],
        #                                              dtype=TF_DTYPE),name='y_init')
        # self.y = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE,name='YY')
        # z_init = tf.Variable(tf.random_uniform([1, self._dim],
        #                                        minval=-.1, maxval=.1,
        #                                        dtype=TF_DTYPE),name='z_init')
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
        # self.y = tf.multiply(self.y,tf.random_uniform([1],
        #                                              minval=self._config.y_init_range[0],
        #                                              maxval=self._config.y_init_range[1],
        #                                              dtype=TF_DTYPE))
        # z = tf.matmul(all_one_vec, z_init)
        self.k = tf.Variable(tf.zeros(shape=[self._dim], dtype=TF_DTYPE), name='kk')
        self.newdata1 = tf.Variable(0,dtype=TF_DTYPE,name='nd1')
        self.newdata3 = tf.Variable(0, dtype=TF_DTYPE, name='nd3')
        self.newdata4 = tf.Variable(0, dtype=TF_DTYPE, name='nd4')
        for p in range(0,self._config.valid_size):
            print(p)
            z = tf.Variable(tf.random_uniform([self._dim],minval=-.1, maxval=.1,dtype=TF_DTYPE),name='zz')
            self.y = tf.random_uniform([1],minval=self._config.y_init_range[0],maxval=self._config.y_init_range[1],dtype=TF_DTYPE)
            for t in range(0, self._num_time_interval-1):
                self.k=tf.assign(self.k,self._x[p, :, t+1])
                self.y=tf.subtract( self.y,tf.multiply(self._bsde.delta_t,self._bsde.f_tf(self._t[t], self._x[p, :, t], self.y , z)
                )) + tf.reduce_sum(tf.multiply(z , self._dw[p, :, t]))
                z = self._subnetwork(self.k, str(t + 1))
                if t==self._num_time_interval-2:
                    self.kk = 0
                    self._hessian = [tf.gradients(z[i], self.k) for i in range(self._dim)]/np.sqrt(2.0)##############
                    for i in range(self._dim):
                        self.kk = self.kk+self._hessian[i,0][i]
                        # self.pri = tf.print(self.kk)
                    self.newdata1=self.newdata1+ self.kk
            self.y  = tf.subtract(self.y,tf.multiply( self._bsde.delta_t , self._bsde.f_tf(
                self._t[-1], self._x[p, :, -2], self.y , z
            ))) + tf.reduce_sum(tf.multiply(z, self._dw[p, :, -1]))
            self.newdata3=self.newdata3+self._bsde.f_tf(self._t[-1], self._x[p, :, -2], self.y , z)
            self.newdata4 = self.newdata4+tf.subtract(self.y,self._bsde.g_tf(self._total_time, self._x[:, :, -1]))
        self.newdata1 = self.newdata1 / self._config.valid_size
        self.newdata3 = self.newdata3 / self._config.valid_size
        self.newdelta=self.newdata3
        # print(self.newloss)
        #########################
        delta = self.newdata4/self._config.valid_size
        # print(delta)
        self._loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                                2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        self.newloss = tf.reduce_mean(tf.where(tf.abs(self.newdelta) < DELTA_CLIP, tf.square( self.newdelta),
                                                2 * DELTA_CLIP * tf.abs( self.newdelta) - DELTA_CLIP ** 2))
        # print(self.newloss.shape)
        # self.pri=tf.print(self.newdelta,['self.newdelta_value: ',self.newdelta])
        self._newloss = self.newloss+self._loss
        # print(self._newloss)
        # train operations
        global_step = tf.get_variable('global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self._config.lr_boundaries,
                                                    self._config.lr_values)
        self.trainable_variables = tf.trainable_variables()
        # print(self.trainable_variables)
        #self.test = tf.gradients(self.y,self._t)
        grads = tf.gradients(self._newloss, self.trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, self.trainable_variables),
                                            global_step=global_step, name='train_step')
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time

    def _subnetwork(self, x, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = self._batch_norm(x, name='path_input_norm')
            for i in range(1, len(self._config.num_hiddens)-1):
                hiddens = self._dense_batch_layer(hiddens,
                                                  self._config.num_hiddens[i],
                                                  activation_fn=tf.nn.relu,
                                                  name='layer_{}'.format(i))
            output = self._dense_batch_layer(hiddens,
                                             self._config.num_hiddens[-1],
                                             activation_fn=None,
                                             name='final_layer')
        return output

    def _dense_batch_layer(self, input_, output_size, activation_fn=None,
                           stddev=5.0, name='linear'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            shape = input_.get_shape().as_list()
            weight = tf.get_variable('Matrix', [shape[0], output_size], TF_DTYPE,
                                     tf.random_normal_initializer(
                                         stddev=stddev/np.sqrt(shape[0]+output_size)))
            hiddens = tf.matmul(tf.reshape(input_ ,[1,shape[0]]), weight)
            hiddens = tf.reshape(hiddens,[output_size])
            hiddens_bn = self._batch_norm(hiddens)
        if activation_fn:
            return activation_fn(hiddens_bn)
        else:
            return hiddens_bn

    def _batch_norm(self, x, affine=True, name='batch_norm'):
        """Batch normalization"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, TF_DTYPE,
                                   initializer=tf.random_normal_initializer(
                                       0.0, stddev=0.1, dtype=TF_DTYPE))
            gamma = tf.get_variable('gamma', params_shape, TF_DTYPE,
                                    initializer=tf.random_uniform_initializer(
                                        0.1, 0.5, dtype=TF_DTYPE))
            moving_mean = tf.get_variable('moving_mean', params_shape, TF_DTYPE,
                                          initializer=tf.constant_initializer(0.0, TF_DTYPE),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, TF_DTYPE,
                                              initializer=tf.constant_initializer(1.0, TF_DTYPE),
                                              trainable=False)
            # These ops will only be preformed when training
            mean, variance = tf.nn.moments(x, [0], name='moments')
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_mean, mean, MOMENTUM))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_variance, variance, MOMENTUM))
            mean, variance = tf.cond(self._is_training,
                                     lambda: (mean, variance),
                                     lambda: (moving_mean, moving_variance))
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, EPSILON)
            y.set_shape(x.get_shape())
            return y
