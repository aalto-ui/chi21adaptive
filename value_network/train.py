#!/usr/bin/env python3
# coding: utf-8

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    y1, y2, y3 = [], [], []

    with open(filepath) as f:
        for line in f.read().splitlines():
            (serial, forage, recall), (target_menu, diff_freq, diff_asso), exposed = format_row(line)

            X1.append(target_menu)
            X2.append(diff_freq)
            X3.append(diff_asso)
            X4.append(exposed)

            y1.append(serial)
            y2.append(forage)
            y3.append(recall)

    return (np.array(X1), np.array(X2), np.array(X3), np.array(X4)), (np.array(y1), np.array(y2), np.array(y3))


def format_row(line):
    (serial, forage, recall), (source_menu, source_freq, source_asso), (target_menu, target_freq, target_asso), exposed = parse_row(line)

    adap_menu, diff_freq, diff_asso = parse_user_input(source_menu, source_freq, source_asso, target_menu, target_freq, target_asso)

    return (serial, forage, recall), (adap_menu, diff_freq, diff_asso), exposed


def parse_row(line):
    # Row format is "[serial,forage,recall][source_menu][source_freq][source_asso][target_menu][target_freq][target_asso][exposed]"
    tokens = line[1:-1].split('][')
    n_toks = len(tokens)
    assert n_toks == 8, 'There are {} tokens, but I expected 8'.format(n_toks)

    # FIXME: We should agree on a parser-friendly row format.
    serial, forage, recall = list(map(float, tokens[0].split(', ')))

    source_menu = list(map(lambda x: x.replace("'", ''), tokens[1].split(', ')))
    source_freq = list(map(float, tokens[2].split(', ')))
    source_asso = list(map(float, tokens[3].split(', ')))

    target_menu = list(map(lambda x: x.replace("'", ''), tokens[4].split(', ')))
    target_freq = list(map(float, tokens[5].split(', ')))
    target_asso = list(map(float, tokens[6].split(', ')))

    # Currently there's only one extra feat, but wrap it as list in case we add more feats in the future.
    exposed = [bool(tokens[7])]

    return (serial, forage, recall), (source_menu, source_freq, source_asso), (target_menu, target_freq, target_asso), exposed


def parse_user_input(source_menu, source_freq, source_asso, target_menu, target_freq, target_asso):
    # Encode adapted menu as integers and compute the difference between previous and current menu configuration.
    adap_menu = onehot_menu(target_menu)
    # Adjust remaining menu items with zeros (reserved value) at the end.
    adap_menu = adj(adap_menu, value=[0])

#    # Experimental: ignore differences w.r.t source menu.
#    num_cols = len(target_freq)
#    tgt_asso = np.array(target_asso).reshape((num_cols, num_cols))
#    tgt_asso = adj([adj(item) for item in tgt_asso], [0]*MAX_MENU_ITEMS)
#    tgt_asso = tgt_asso.reshape((MAX_MENU_ITEMS*MAX_MENU_ITEMS,))
#    return adap_menu, adj(target_freq), tgt_asso

    # Ensure that all vectors have the same length.
    max_freq_len = max(len(source_freq), len(target_freq))
    max_asso_len = max(len(source_asso), len(target_asso))
    source_freq = pad(source_freq, max_freq_len)
    target_freq = pad(target_freq, max_freq_len)
    source_asso = pad(source_asso, max_asso_len)
    target_asso = pad(target_asso, max_asso_len)

    diff_freq = np.diff([source_freq, target_freq], axis=0).flatten()
    diff_asso = np.diff([source_asso, target_asso], axis=0).flatten()

    # Ensure there is a change in freq distribution, otherwise `diff_freq` would be always zero.
    if np.array_equal(source_freq, target_freq):
        diff_freq = source_freq

    # The association matrix list is given as a flat vector, so reshape it before padding.
    # Notice that we read the number of items BEFORE padding `diff_freq`.
    num_rows = len(diff_freq)
    num_cols = len(diff_asso)//num_rows
    diff_asso = diff_asso.reshape((num_cols, num_rows))
    diff_asso = adj([adj(item) for item in diff_asso], [0]*MAX_MENU_ITEMS)
    diff_asso = diff_asso.reshape((MAX_MENU_ITEMS*MAX_MENU_ITEMS,))

    return adap_menu, adj(diff_freq), diff_asso


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


def create_model(adap_menu, diff_freq, diff_asso, xtra_feat):
    # The provided sample args are needed to get the input shapes right.
    # For example, the network capacity is bounded by the (max) number of menu items.
    num_items = diff_freq.shape[0]

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

    def serial_tail(inputs):
        s = tf.keras.layers.Dense(num_items//2)(inputs)
        s = tf.keras.layers.Dropout(0.5)(s)
        s = tf.keras.layers.Dense(1)(s)
        s = tf.keras.layers.Activation('linear', name='serial_output')(s)
        return s

    def forage_tail(inputs):
        f = tf.keras.layers.Dense(num_items//2)(inputs)
        f = tf.keras.layers.Dropout(0.5)(f)
        f = tf.keras.layers.Dense(1)(f)
        f = tf.keras.layers.Activation('linear', name='forage_output')(f)
        return f

    def recall_tail(inputs):
        r = tf.keras.layers.Dense(num_items//2)(inputs)
        r = tf.keras.layers.Dropout(0.5)(r)
        r = tf.keras.layers.Dense(1)(r)
        r = tf.keras.layers.Activation('linear', name='recall_output')(r)
        return r

    input_menu = tf.keras.layers.Input(shape=adap_menu.shape, name='menu')
    input_freq = tf.keras.layers.Input(shape=diff_freq.shape, name='priors')
    input_asso = tf.keras.layers.Input(shape=diff_asso.shape, name='associations')
    input_feat = tf.keras.layers.Input(shape=xtra_feat.shape, name='features')

    menu = menu_head(input_menu)
    freq = freq_head(input_freq)
    asso = asso_head(input_asso)

    combined_head = tf.keras.layers.concatenate([menu.output, freq.output, asso.output, input_feat])
    serial = serial_tail(combined_head)
    forage = forage_tail(combined_head)
    recall = recall_tail(combined_head)

    # Hereby I compose the almighty value network model.
    model = tf.keras.Model(inputs=[menu.input, freq.input, asso.input, input_feat], outputs=[serial, forage, recall])
    losses = {'serial_output': 'mse', 'forage_output': 'mse', 'recall_output': 'mse'}
    model.compile(optimizer='rmsprop', loss=losses, metrics=['mse', 'mae'])

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
    y1, y2, y3 = [], [], []

    for f in tr_files:
        (X1_, X2_, X3_, X4_), (y1_, y2_, y3_) = load_data(file_path)
        X1 = np.concatenate((X1, X1_)) if len(X1) > 0 else X1_
        X2 = np.concatenate((X2, X2_)) if len(X2) > 0 else X2_
        X3 = np.concatenate((X3, X3_)) if len(X3) > 0 else X3_
        X4 = np.concatenate((X4, X4_)) if len(X4) > 0 else X4_
        y1 = np.concatenate((y1, y1_)) if len(y1) > 0 else y1_
        y2 = np.concatenate((y2, y2_)) if len(y2) > 0 else y2_
        y3 = np.concatenate((y3, y3_)) if len(y3) > 0 else y3_

    # Provide one sample of the input data to the model.
    model = create_model(X1[0], X2[0], X3[0], X4[0])

#    model.summary()
#    tf.keras.utils.plot_model(model, show_shapes=False, to_file='value_network.png')
#    tf.keras.utils.plot_model(model, show_shapes=True, to_file='value_network_with_shapes.png')
#    tf.keras.utils.plot_model(model, show_shapes=False, show_layer_names=False, to_file='value_network_blocks.png')

    from time import time
    now = int(time())

    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir='./training_logs_{}'.format(now)),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    ]

    model.fit([X1, X2, X3, X4], [y1, y2, y3], validation_split=0.2, epochs=200, batch_size=32, callbacks=cbs)
    model.save('value_network.h5')
