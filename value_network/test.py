#!/usr/bin/env python3
# coding: utf-8

import sys
import random

from train import parse_row
from model import ValueNetwork


model_file = sys.argv[1] # e.g. value_network_5items.h5
vn = ValueNetwork(model_file)

# The test dataset must follow the same format of the training data file,
# even though we ignore the labels while testing.
datafile = sys.argv[2]

samples = []
with open(datafile) as f:
    lines = f.read().splitlines()
    # Grab a random sample of the data.
    random.shuffle(lines)
    for line in lines[0:10]:
        _, (source_menu, source_freq, source_asso), (target_menu, target_freq, target_asso), exposed = parse_row(line)
        samples.append([source_menu, source_freq, source_asso, target_menu, target_freq, target_asso, exposed])

results = vn.predict_batch(samples)
print('Num results', len(results))
print(results[0])

