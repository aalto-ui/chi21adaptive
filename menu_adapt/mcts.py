# MCTS Implementation. During rollouts, the user oracle is used to predict rewards for adaptations

from __future__ import division, print_function
import time
import math
import random
import sys
import utility
import os
from useroracle import UserOracle
from copy import deepcopy
from adaptation import Adaptation
from state import AdaptationType
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'value_network'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'policy_network'))
from value_network_model import ValueNetwork


# Rollout policy: random
def random_policy(state, oracle):
    rewards = [0.0,0.0,0.0]
    # if state.exposed: rewards = oracle.get_individual_rewards(state)[0]
    while not oracle.is_terminal(state):
        try:
            adaptation = random.choice(state.menu_state.possible_adaptations())
        except IndexError:
            raise Exception("Non-terminal state has no possible adaptations: " + str(state))
        state = state.take_adaptation(adaptation)
        if state.exposed:
            new_rewards = oracle.get_individual_rewards(state)[0]
            rewards = [a + b for a,b in zip(rewards, new_rewards)]                
    return rewards

# MCTS node
class TreeNode():
    def __init__(self, state, parent):
        self.state = state # Menu now
        self.parent = parent
        self.num_visits = 0 # For tracking n in UCT
        self.total_rewards = [0.0,0.0,0.0] # For tracking q in UCT
        self.children = {}
        self.fully_expanded = False # Is it expanded already?

    def __str__(self):
        return str(self.state) + "," + str(self.total_rewards)

# MCTS tree
class mcts():
    def __init__(self, useroracle, weights, objective, use_network, network_name = None, limit_type = 'time', time_limit=None, num_iterations=None, exploration_const=1.0/math.sqrt(2),
                 rollout_policy=random_policy):
        
        self.oracle = useroracle # User oracle used
        self.objective = objective # Average, Conservative, or optimistic objective - used to compute total reward
        self.weights = weights  # Weights for combining the 3 strategies when using the "average" objective
        self.time_limit = time_limit # Time limit to search
        self.limit_type = limit_type # Type of computation budget
        self.num_iterations = num_iterations # No. of iterations to run
        self.exploration_const = exploration_const # Original exploration constant: 1 / math.sqrt(2)
        self.rollout = rollout_policy # Rollout policy used
        self.use_network = use_network
        if self.use_network and network_name:
            self.vn = ValueNetwork("networks/"+network_name)

        
    def __str__(self):
        tree_str = str(self.root) + "\n"
        for child in self.root.children.values():
            tree_str += str(child) + "\n"
        return tree_str
    
    def execute_round(self):
        node = self.select_node(self.root)        
        if node is not self.root and self.use_network:
            rewards = self.get_reward_predictions(node)
        else: 
            rewards = self.rollout(node.state, self.oracle)
        self.backpropagate(node, rewards)

    def search(self, initial_state, initial_node = None):
        if initial_node: 
            self.root = initial_node
            self.root.parent = None
        else: self.root = TreeNode(initial_state, None)
        time_limit = time.time() + self.time_limit / 1000
        if self.limit_type == 'time':
            while time.time() < time_limit:
                self.execute_round()            
        elif self.limit_type == 'iterations':
            for _ in self.num_iterations:
                self.execute_round()

        adaptation_probability = self.get_adaptation_probabilities(self.root, 0.0)
        best_child = self.get_best_child(self.root, 0.0)
        best_adaptation = self.get_adaptation(self.root, best_child)
        avg_rewards = [x/best_child.num_visits for x in best_child.total_rewards]
        
        return best_adaptation, best_child, avg_rewards, adaptation_probability


    def get_reward_predictions(self, node):
        rewards = [0.0,0.0,0.0]
        if node.parent is not None:
            samples = []
            target_menu = node.state.menu_state.simplified_menu(trailing_separators=True)
            source_menu = node.parent.state.menu_state.simplified_menu(trailing_separators=True)
            target_state = node.state
            source_state = node.parent.state
            source_assoc = utility.get_association_matrix(source_menu, source_state.menu_state.associations)
            source_freq = utility.get_sorted_frequencies(source_menu, source_state.user_state.freqdist)
            target_assoc = utility.get_association_matrix(target_menu, target_state.menu_state.associations)
            target_freq = utility.get_sorted_frequencies(target_menu, target_state.user_state.freqdist)
            exposed = node.state.exposed
            samples.append([source_menu,source_freq,source_assoc,target_menu,target_freq,target_assoc,[bool(exposed)]])
            predictions = self.vn.predict_batch([samples[0]]) # Get predictions from value network (if usenetwork is true)
            rewards = predictions[0]
        return rewards

        
    def select_node(self, node):
        while not self.oracle.is_terminal(node.state):
            if node.fully_expanded:
                node = self.get_best_child(node, self.exploration_const)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        adaptations = node.state.menu_state.possible_adaptations()
        #Always try the "do nothing path first"
        if adaptations[-1] not in node.children.keys():
            adaptation = adaptations[-1]
            newNode = TreeNode(node.state.take_adaptation(adaptation), node)
            node.children[adaptation] = newNode
            return newNode

        random.shuffle(adaptations)
        for  adaptation in adaptations:
            if adaptation not in node.children.keys():
                newNode = TreeNode(node.state.take_adaptation(adaptation), node)
                node.children[adaptation] = newNode
                if len(adaptations) == len(node.children) or self.oracle.is_terminal(newNode.state):
                    node.fully_expanded = True
                return newNode
        raise Exception("Ouch! Should never reach here")

    def backpropagate(self, node, rewards):
        while node is not None:
            node.num_visits += 1
            node.total_rewards = [a+b for a,b in zip(node.total_rewards,rewards)]
            node = node.parent

    # Pick best child as next state
    def get_best_child(self, node, exploration_const):
        best_value = float("-inf")
        best_node = None
        # return argmax(customFunction(node, frequencies, associations))
        children = list(node.children.values())
        random.shuffle(children)
        for child in children:
            # node value using UCT
            total_reward = self.compute_reward(child.total_rewards)
            node_value = total_reward/child.num_visits + exploration_const * math.sqrt(math.log(node.num_visits) / child.num_visits)
            
            if node_value > best_value:
                best_value = node_value
                best_node = child

        return best_node

    def compute_reward(self,total_rewards):
        if self.objective == "AVERAGE":
            total_reward = sum([a*b for a,b in zip(self.weights, total_rewards)]) # Take average reward 
        elif self.objective == "OPTIMISTIC":
            total_reward = max(total_rewards) # Take best reward
        elif self.objective == "CONSERVATIVE":
            total_reward = min(total_rewards) if min(total_rewards) >= 0 else min(total_rewards)*2 # Take minimum; add penalty if negative
        return total_reward


    def get_adaptation(self, root, best_child):
        for adaptation, node in root.children.items():
            if node is best_child:
                return adaptation
                
    def get_adaptation_probabilities(self, node, exploration_const):
        if node.children == 0: return None
        # Transition probability for children. Dict. Key = adaptation; Value = probability
        probability = {a:0.0 for a in node.state.menu_state.possible_adaptations()}
        for adaptation,child in node.children.items():
            probability[adaptation] = child.num_visits/node.num_visits
        return probability
        
       
    def get_best_adaptation(self, root):
        best_num_visits = 0
        best_results = {}
        for adaptation,child in root.children.items():
            if child.num_visits > best_num_visits:
              best_num_visits = child.num_visits
              best_results = {adaptation:child}
            elif child.num_visits == best_num_visits:
              best_num_visits = child.num_visits
              best_results[adaptation] = child
        
        best_adaptation, best_child = random.choice(list(best_results.items())) 
              
        return best_adaptation, best_child