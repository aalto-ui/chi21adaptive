#!/usr/bin/env python3
# coding: utf-8

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import numpy as np
import tensorflow as tf


# Boost GPU usage.
conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
tf.compat.v1.Session(config=conf)

# Maximum number of items in a given menu, including separators.
MAX_MENU_ITEMS = 20
# Size of the one-hot encoded vectors. This value should be large enough to avoid hashing collisions.
ENC_VOCAB_SIZE = 90


def load_data(filepath):
    X1, X2, X3, X4 = [], [], [], []
    y1 = []

    with open(filepath) as f:
        for line in f.read().splitlines():
            (source_menu, source_freq, source_asso), exposed, adaptations = format_row(line)

            for state, prob in adaptations.items():
                X1.append(source_menu)
                X2.append(source_freq)
                X3.append(source_asso)
                X4.append(tuple(exposed) + state)

                y1.append(prob)

    return (np.array(X1), np.array(X2), np.array(X3), np.array(X4)), np.array(y1)


def format_row(line):
    (source_menu, source_freq, source_asso), exposed, adaptations = parse_row(line)

    src_menu, src_freq, src_asso = parse_user_input(source_menu, source_freq, source_asso)

    return (src_menu, src_freq, src_asso), exposed, adaptations


def parse_row(line):
    # Row format is "[source_menu][source_freq][source_asso][exposed]{state_dict}"
    # Notice that there's no target here, since each adaptation results in a different target menu.
    stuff, state_dict = line.split(']{')

    # FIXME: We should agree on a parser-friendly row format.
    state_dict = '{' + state_dict.replace("'", '"')
    state_dict = json.loads(state_dict)

    tokens = stuff[1:].split('][')
    n_toks = len(tokens) 
    assert n_toks == 4, 'There are {} tokens, but I expected 4'.format(n_toks)

    source_menu = list(map(lambda x: x.replace("'", ''), tokens[0].split(', ')))
    source_freq = list(map(float, tokens[1].split(', ')))
    source_asso = list(map(float, tokens[2].split(', ')))

    # Allocate all possible state from this menu, including non-feasible ones.
    adaptations = state_allocator(state_dict, len(source_menu))

    # Currently there's only one extra feat, but wrap it as list in case we add more feats in the future.
    exposed = [bool(tokens[3])]

    return (source_menu, source_freq, source_asso), exposed, adaptations


def state_allocator(dic, menu_len):
    res = {}
    for a in range(menu_len):
        for b in range(menu_len):
            for c in range(3):
                for d in range(1):
                    tup = (a,b,c,d)
                    key = str(tup)
                    # Unfeasible states get zero probability.
                    res[tup] = dic[key] if key in dic else 0
    return res


def parse_user_input(source_menu, source_freq, source_asso):
    source_menu = onehot_menu(source_menu)
    # Adjust remaining menu items with zeros (reserved value) at the end.
    source_menu = adj(source_menu, value=[0])

    # Ensure that all vectors have the same length.
    source_freq = np.array(pad(source_freq, len(source_freq)))
    source_asso = np.array(pad(source_asso, len(source_asso)))

    # The association matrix list is given as a flat vector, so reshape it before padding.
    # Notice that we read the number of items BEFORE padding `source_freq`.
    num_rows = len(source_freq)
    num_cols = len(source_asso)//num_rows
    source_asso = source_asso.reshape((num_cols, num_rows))
    source_asso = adj([adj(item) for item in source_asso], [0]*MAX_MENU_ITEMS)
    source_asso = source_asso.reshape((MAX_MENU_ITEMS*MAX_MENU_ITEMS,))

    return source_menu, adj(source_freq), source_asso


def pad(l, size, value=0):
    return l + [value] * abs((len(l)-size))


def adj(vec, value=0):
    N = len(vec)
    d = MAX_MENU_ITEMS - N
    if d < 0:
        # Truncate vector.
        vec = vec[:MAX_MENU_ITEMS]
    elif d > 0:
        # Pad vector with zeros (reserved value) at the *end* of the vector.
        vec = list(vec) + [value for _ in range(d)]
    return np.array(vec)


def onehot_menu(items):
    # FIXME: We should agree on a single-word menu separator, because '----' is conflicting with the built-in text parser.
    enc_menu = [tf.keras.preprocessing.text.one_hot(w, ENC_VOCAB_SIZE, filters='') for w in items]
    return enc_menu


def create_model(source_menu, source_freq, source_asso, source_feat):
    # The provided sample args are needed to get the input shapes right.
    # For example, the network capacity is bounded by the (max) number of menu items.
    num_items = source_freq.shape[0]

    def menu_head(inputs):
        m = tf.keras.layers.Embedding(ENC_VOCAB_SIZE, num_items, input_length=num_items)(inputs)
        m = tf.keras.layers.Flatten()(m)
        m = tf.keras.layers.Dropout(0.5)(m)
        m = tf.keras.layers.Dense(num_items//2)(m)
        m = tf.keras.Model(inputs=inputs, outputs=m)
        return m

    def freq_head(inputs):
        f = tf.keras.layers.Reshape((num_items, 1))(inputs)
        f = tf.keras.layers.LSTM(num_items, activation='relu')(f)
        f = tf.keras.layers.Dropout(0.5)(f)
        f = tf.keras.layers.Dense(num_items//2)(f)
        f = tf.keras.Model(inputs=inputs, outputs=f)
        return f

    def asso_head(inputs):
        a = tf.keras.layers.Reshape((num_items, num_items))(inputs)
        a = tf.keras.layers.LSTM(num_items*2, activation='relu')(a)
        a = tf.keras.layers.Dropout(0.5)(a)
        a = tf.keras.layers.Dense(num_items//2)(a)
        a = tf.keras.Model(inputs=inputs, outputs=a)
        return a

    def prob_tail(inputs):
        s = tf.keras.layers.Dense(num_items//2)(inputs)
        s = tf.keras.layers.Dropout(0.5)(s)
        s = tf.keras.layers.Dense(1)(s)
        s = tf.keras.layers.Activation('sigmoid', name='state_output')(s)
        return s

    input_menu = tf.keras.layers.Input(shape=source_menu.shape, name='menu')
    input_freq = tf.keras.layers.Input(shape=source_freq.shape, name='priors')
    input_asso = tf.keras.layers.Input(shape=source_asso.shape, name='associations')
    input_feat = tf.keras.layers.Input(shape=source_feat.shape, name='features')

    menu = menu_head(input_menu)
    freq = freq_head(input_freq)
    asso = asso_head(input_asso)

    combined_head = tf.keras.layers.concatenate([menu.output, freq.output, asso.output, input_feat])
    state_prob = prob_tail(combined_head)

    # Hereby I compose the almighty policy network model.
    model = tf.keras.Model(inputs=[menu.input, freq.input, asso.input, input_feat], outputs=state_prob)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

    return model



if __name__ == '__main__':
    # Input can be either a list of files or a directory.
    train_inputs = sys.argv[1:]

    # Collect all training files first.
    tr_files = []
    for tr_input in train_inputs:
        if os.path.isdir(tr_input):
            for path, directories, files in os.walk(tr_input):
                for f in files:
                    if f.endswith('.txt'):
                        file_path = os.path.join(path, f)
                        tr_files.append(file_path)

        elif os.path.isfile(tr_input):
            tr_files.append(tr_input)

    X1, X2, X3, X4 = [], [], [], []
    y1 = []

    for f in tr_files:
        (X1_, X2_, X3_, X4_), y1_ = load_data(f)
        X1 = np.concatenate((X1, X1_)) if len(X1) > 0 else X1_
        X2 = np.concatenate((X2, X2_)) if len(X2) > 0 else X2_
        X3 = np.concatenate((X3, X3_)) if len(X3) > 0 else X3_
        X4 = np.concatenate((X4, X4_)) if len(X4) > 0 else X4_
        y1 = np.concatenate((y1, y1_)) if len(y1) > 0 else y1_

    # Provide one sample of the input data to the model.
    model = create_model(X1[0], X2[0], X3[0], X4[0])

#    model.summary()
#    tf.keras.utils.plot_model(model, show_shapes=False, to_file='policy_network.png')
#    tf.keras.utils.plot_model(model, show_shapes=True, to_file='policy_network_with_shapes.png')
#    tf.keras.utils.plot_model(model, show_shapes=False, show_layer_names=False, to_file='policy_network_blocks.png')

    from time import time
    now = int(time())

    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir='./training_logs_{}'.format(now)),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    ]

    model.fit([X1, X2, X3, X4], y1, validation_split=0.2, epochs=200, batch_size=32, callbacks=cbs)
    model.save('policy_network.h5')
