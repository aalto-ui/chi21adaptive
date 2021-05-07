#!/usr/bin/env python3
# coding: utf-8

'''Policy network model for MCTS'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
import numpy as np
import tensorflow as tf
from time import time
from policy_train import parse_user_input, state_allocator

np.random.seed(42)
tf.random.set_seed(42)

class PolicyNetwork:

    def __init__(self, modelfile):
        self.model = tf.keras.models.load_model(modelfile)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        

    def predict_batch(self, data):
        '''
        Predict menu states for the given menu data.
        See `predict()` method.
        '''

        predictions = []

        for (source_menu, source_freq, source_asso, exposed) in data:

            X1, X2, X3, X4 = [], [], [], []

            src_menu, src_freq, src_asso = parse_user_input(source_menu, source_freq, source_asso)

            adaptations = state_allocator({}, len(source_menu))
            for state in adaptations:
                X1.append(src_menu)
                X2.append(src_freq)
                X3.append(src_asso)
                X4.append(tuple(exposed) + state)

            X1 = np.array(X1)
            X2 = np.array(X2)
            X3 = np.array(X3)
            X4 = np.array(X4)

#            print('Predicting ...', file=sys.stderr)
#            t_ini = time()

            probs = self.model.predict([X1, X2, X3, X4])

#            t_end = time()
#            print('Elapsed: {} s'.format(t_end - t_ini), file=sys.stderr))

            # Compose adaptations dict, according to the training format.
            state_dict = {}
            for i, state in enumerate(adaptations):
                state_dict[state] = probs[i][0]

            predictions.append(state_dict)

        return predictions


    def predict(self, source_menu, source_priors, source_assocs, exposed):
        '''
        Predict menu states for the proposed menu adaption w.r.t. the original menu.

        Args:
          source_menu (list): Source menu configuration.
          source_priors (list): Click frequency over source menu items.
          source_assocs (list): Association matrix between source menu items.

          exposed (list): Boolean flag indicating that the menu was exposed to the user.

        Returns:
          states (dict): Scores for each possible menu adaptation.
        '''

        warnings.warn('You might want to call predict_batch() instead, since it is MUCH more efficient.')

        return self.predict_batch([
            [source_menu, source_priors, source_assocs, exposed]
        ])
