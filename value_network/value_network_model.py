#!/usr/bin/env python3
# coding: utf-8

'''Value network model for MCTS'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
import numpy as np
import tensorflow as tf
from time import time
from train import parse_user_input

np.random.seed(42)
tf.random.set_seed(42)

class ValueNetwork:

    def __init__(self, modelfile):
        self.model = tf.keras.models.load_model(modelfile)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        

    def predict_batch(self, data):
        '''
        Predict scores and costs for the given menu data.
        See `predict()` method.
        '''

        X1, X2, X3, X4 = [], [], [], []

        for (source_menu, source_freq, source_asso,
             target_menu, target_freq, target_asso, exposed) in data:

            adap_menu, diff_freq, diff_asso = parse_user_input(source_menu, source_freq, source_asso,
                                                               target_menu, target_freq, target_asso)

            X1.append(adap_menu)
            X2.append(diff_freq)
            X3.append(diff_asso)
            X4.append(exposed)

        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X4 = np.array(X4)

#        print('Predicting ...', file=sys.stderr)
#        t_ini = time()

        (serial, forage, recall) = self.model.predict([X1, X2, X3, X4])

#        t_end = time()
#        print('Elapsed: {} s'.format(t_end - t_ini), file=sys.stderr))

        # Format output row-wise instead of batch-wise.
        output = []
        for i, _ in enumerate(serial):
            sr = serial[i][0]
            fr = forage[i][0]
            rr = recall[i][0]
            output.append([sr, fr, rr])
        return output


    def predict(self, source_menu, source_priors, source_assocs,
                      target_menu, target_priors, target_assocs, exposed):
        '''
        Predict rewards for the proposed menu adaption w.r.t. the original menu.

        Args:
          source_menu (list): Source menu configuration.
          source_priors (list): Click frequency over source menu items.
          source_assocs (list): Association matrix between source menu items.

          target_menu (list): Target menu configuration.
          target_priors (list): Click frequency over target menu items.
          target_assocs (list): Association matrix between target menu items.

          exposed (list): Boolean flag indicating that the menu was exposed to the user.

        Returns:
          serial (float): Serial reward
          forage (float): Forage reward
          recall (float): Recall reward
        '''

        warnings.warn('You might want to call predict_batch() instead, since it is MUCH more efficient.')

        return self.predict_batch([
            [source_menu, source_priors, source_assocs,
             target_menu, target_priors, target_assocs, exposed]
        ])
