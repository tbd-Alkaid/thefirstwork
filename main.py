"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""
from __future__ import print_function
import json
import logging
import os
import numpy as np
import tensorflow as tf
from config import get_config
from equation import get_equation
from solver import FeedForwardModel
from tensorflow.python import debug as tf_debug

def main():
    problem_name0 = 'AllenCahn'
    config = get_config(problem_name0)
    bsde = get_equation(problem_name0, config.dim, config.total_time, config.num_time_interval)
    log_dir = './logs'
    if not os.path.exists(log_dir):
         os.mkdir(log_dir)
    path_prefix = os.path.join(log_dir, problem_name0)
    # with open('{}_config.json'.format(path_prefix), 'w') as outfile:
    #     json.dump(dict((name, getattr(config, name))
    #                    for name in dir(config) if not name.startswith('__')),
    #               outfile, indent=2)
    # logging.basicConfig(level=logging.INFO, b
    #                     format='%(levelname)-6s %(message)s')

    for idx_run in range(1, 2):
        tf.reset_default_graph()
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # logging.info('Begin to solve %s with run %d' % (problem_name0, idx_run))
            model = FeedForwardModel(config, bsde, sess)
            # if bsde.y_init:
            #     logging.info('Y0_true: %.4e' % bsde.y_init)
            model.build()
            training_history = model.train()
            # training_history.reshape(1,-1)
            print(training_history)
            np.savetxt('{}_training_history_{}.csv'.format(path_prefix, idx_run),
                       training_history,
                       fmt=['%.5e','%d'],
                       delimiter=",",
                       header="loss_function,elapsed_time",
                       comments='')
            writer = tf.summary.FileWriter('./graph/qgraph',sess.graph)

if __name__ == '__main__':
    main()
