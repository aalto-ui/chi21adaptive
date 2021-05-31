# Adapting User Interfaces with Model-based Reinforcement Learning (ACM CHI 2021)
### By Kashyap Todi, Gilles Bailly, Luis A. Leiva, Antti Oulasvirta
### https://userinterfaces.aalto.fi/adaptive
#### Copyright (c) 2021 Aalto University. All rights reserved.

This code repository is for the adaptive menus application described in the CHI 2021 paper: https://userinterfaces.aalto.fi/adaptive/resources/chi2021-todi-adaptive.pdf

## Requirements
1. Ray (https://ray.io) for parallelisation.
2. TensorFlow 2 (https://www.tensorflow.org/install/pip) for using the neural networks.

## Key Components for the Menu Adaptation Application

* `plan.py` is the starting point for code execution. To generate results, execute:

* ```python3 plan.py```
The command will run the MCTS planner for the 5-item case (`menu_5items.txt`) without the value network. To use the value network, add the `-nn` option.

* ```python3 plan.py -h```
See the full list of options available for running `plan.py`

* `utility.py` contains useful functions for loading data, initialisation, etc.

* Input files are stored within `Input` folder. For each case, there's an input menu, association list, and user history.

* `state.py` defines the menu and user state. The root state is initialised using the input menu, associations, and user history

* `adaptation.py` provides a general format for defining adaptations. It uses the syntax `(i,j,type,expose)` where `i` and `j` are two positions in the menu, `type` specifies the type of adaptation (e.g. swap, move, group move), and `expose` is a boolean that specifies whether the adapted menu is exposed to the user or not.

* `mcts.py` contains the code for Monte Carlo tree search.

* `useroracle.py` defines the user models for running simulations. These models are used towards predicting task completion time given a menu design, and for computing the reward after making an adaptation.

[Read the Value Network documentation.](./value_network/README.md)

[Read the Policy Network documentation.](./policy_network/README.md)
