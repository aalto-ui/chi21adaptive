#!/usr/bin/env python3
# coding: utf-8

import sys
import random

from policy_train import parse_row
from model import PolicyNetwork


model_file = sys.argv[1] # e.g. policy_network_5items.h5
pn = PolicyNetwork(model_file)

# The test dataset must follow the same format of the training data file,
# even though we ignore the labels while testing.
datafile = sys.argv[2]

samples = []
with open(datafile) as f:
    lines = f.read().splitlines()
    # Grab a random sample of the data.
    random.shuffle(lines)
    for line in lines[0:10]:
        (source_menu, source_freq, source_asso), exposed, _ = parse_row(line)
        samples.append([source_menu, source_freq, source_asso, exposed])

results = pn.predict_batch(samples)
print('Num results', len(results))
print(results[0])

