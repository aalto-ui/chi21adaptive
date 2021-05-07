from copy import deepcopy
import operator
import utility
from enum import Enum
from adaptation import Adaptation
import math
class AdaptationType(Enum):
    NONE = 0
    MOVE = 1
    SWAP = 2
    GROUP_MOVE = 3
    GROUP_SWAP = 4
    ADD_SEP = 5
    REMOVE_SEP = 6
    MOVE_SEP = 7

    def __str__ (self):
        return self.name
    
    def __repr__(self):
        return self.name

class State():
    separator = "----"
    number_of_clicks = 20
    # Initialise the state
    def __init__(self, menu_state, user_state, previous_seen_state = None, depth = 0, exposed = False):
        self.user_state = user_state
        self.menu_state = menu_state
        self.previous_seen_state = previous_seen_state
        self.depth = depth
        self.exposed = exposed

    # Function called when an adaptation is made. The user state and menu state are updated accordingly
    def take_adaptation(self, adaptation, update_user = True):        
        new_state = deepcopy(self)
        new_state.depth += 1
        new_state.exposed = adaptation.expose
        if self.exposed: new_state.previous_seen_state = self

        # Simulate the next user session by adding clicks
        if self.exposed and update_user:
            new_state.user_state.update(menu = self.menu_state.menu, number_of_clicks = self.number_of_clicks)
        # Adapt the menu

        new_state.menu_state.menu = self.menu_state.adapt_menu(adaptation)
        
        return new_state


# Defines the menu - includes the list of menu items, and association list for each item. 
class MenuState():
    separator = "----"
    def __init__(self, menu, associations):
        self.menu = menu
        self.associations = associations
        separatorsalready = menu.count(self.separator) # How many separators we have?
        # max_separators = int(min(math.ceil(len(self.menu)/1.5), 8))
        max_separators = 4
        if separatorsalready < max_separators:
            for _ in range (separatorsalready, max_separators): # Append the remaining of separators
                self.menu.append(self.separator)


    def __str__(self):
        return str(self.simplified_menu())

    def __repr__(self):
        return str(self.simplified_menu())
    
    # Returns list of all adaptations that are feasible from the menu state.
    def possible_adaptations(self):
        
        possibleadaptations = []
        seen_menus = [self.simplified_menu()]
        max_distance = 3

        simple_menu = list(filter(("----").__ne__, self.menu)) # menu without separators
            
        # swaps
        for i in range (0, len(self.menu)):
            for j in range (i+1, len(self.menu)):
                if self.menu[i] == self.separator and self.menu[j] == self.separator: continue # no point swapping separators
                if len(simple_menu)>10:
                    if abs(i-j)>max_distance and self.menu[i] is not self.separator: 
                        continue # Limit max swap distance
                test_adaptation = Adaptation([i,j,AdaptationType.SWAP, True])
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)
                if (adapted_menu.simplified_menu() not in seen_menus):
                    seen_menus.append(adapted_menu.simplified_menu())
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.SWAP, True]))
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.SWAP, False]))

        # moves
        for i in range (0, len(self.menu)):
            for j in range (0, len(self.menu)):
                if i == j or (i == j+1): continue
                if len(simple_menu) > 10:
                    if self.menu[i] != self.separator and abs(j-i) > max_distance: 
                        continue # Limit max move distance for large menus
                if self.menu[i] == self.separator:
                    if j != len(self.menu)-1 and (self.menu[j] == self.separator or self.menu[j-1] == self.separator):
                        continue   
                test_adaptation = Adaptation([i,j,AdaptationType.MOVE, True])
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)
                if (adapted_menu.simplified_menu() not in seen_menus):
                    seen_menus.append(adapted_menu.simplified_menu())                     
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.MOVE, True]))
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.MOVE, False]))
                
        # group adaptations
        menu_string = ";".join(self.menu)
        groups = menu_string.split(self.separator)
        groups = list(filter((";").__ne__, groups))
        groups = list(filter(("").__ne__, groups))
        # Swap groups
        for i in range (0, len(groups)):
            for j in range (i+1, len(groups)):
                test_adaptation = Adaptation([i,j,AdaptationType.GROUP_SWAP, True])
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)
                if (adapted_menu.simplified_menu() not in seen_menus):
                    seen_menus.append(adapted_menu.simplified_menu())
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_SWAP, True]))
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_SWAP, False]))
        # Move groups
        for i in range (0, len(groups)):
            for j in range (0, len(groups)):
                if i == j or (i == j+1): continue
                test_adaptation = Adaptation([i,j,AdaptationType.GROUP_MOVE, True])
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)
                if (adapted_menu.simplified_menu() not in seen_menus):
                    seen_menus.append(adapted_menu.simplified_menu())
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_MOVE, True]))
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_MOVE, False]))
        
        # do nothing, show menu
        possibleadaptations.append(Adaptation([0,0,AdaptationType.NONE,True]))
        return possibleadaptations

    # Function to modify the menu by making an adaptation.
    def adapt_menu(self, adaptation):
        new_menu = self.menu.copy()
        if adaptation.type == AdaptationType.SWAP:
            new_menu[adaptation.i], new_menu[adaptation.j] = new_menu[adaptation.j], new_menu[adaptation.i]
        elif adaptation.type == AdaptationType.MOVE:
            del new_menu[adaptation.i]
            new_menu.insert(adaptation.j, self.menu[adaptation.i])
        elif adaptation.type == AdaptationType.GROUP_SWAP or adaptation.type == AdaptationType.GROUP_MOVE:
            menu_string = ";".join(new_menu)
            groups = menu_string.split(self.separator)
            groups = list(filter((";").__ne__, groups))
            groups = list(filter(("").__ne__, groups))
            if adaptation.type == AdaptationType.GROUP_SWAP:
                groups[adaptation.i],groups[adaptation.j] = groups[adaptation.j], groups[adaptation.i]
            elif adaptation.type == AdaptationType.GROUP_MOVE:
                original_groups = groups.copy()
                del groups[adaptation.i]
                groups.insert(adaptation.j, original_groups[adaptation.i])
            groups_string = ";----;".join(groups)
            new_menu = groups_string.split(";")
            new_menu = list(filter("".__ne__,new_menu))
            missing_separators = len(self.menu) - len(new_menu)
            for _ in range (0, missing_separators): # Append the remaining of separators
                new_menu.append(self.separator)
        return new_menu

    # Returns a simplified representation of the menu by ignoring redundant/unnecessary separators
    def simplified_menu(self, trailing_separators = True):
        simplified_menu = []
        for i in range (0,len(self.menu)):
            if self.menu[i] != self.separator: 
                simplified_menu.append(self.menu[i])
                continue
            if self.menu[i] == self.separator and len(simplified_menu)>0:
                if simplified_menu[-1] == self.separator: continue
                simplified_menu.append(self.menu[i])

        if simplified_menu[0] == self.separator:
                del simplified_menu[0]
        if simplified_menu[-1] == self.separator:
                del simplified_menu[-1]   
        if trailing_separators:
            old_length = len(self.menu)
            new_length = len(simplified_menu)
            sep_to_add = old_length - new_length
            for _ in range (sep_to_add): # Append the remaining of separators
                simplified_menu.append(self.separator)

        return simplified_menu
            
# Defines the user's interest and expertise
class UserState():
    def __init__(self, freqdist, total_clicks, history):
        self.freqdist = freqdist # Normalised click frequency distribution (position-independent)
        self.total_clicks = total_clicks
        self.history = history
        item_history = [row[0] for row in self.history]

        self.recall_practice = {} # Count of clicks at last-seen position (resets when position changes)
        self.activations = self.get_activations()
        for key,_ in self.freqdist.items():
            self.recall_practice[key] = item_history.count(key)
        if int(total_clicks) != len(self.history): print("HMM something wrong with the history")
        # for i in range(0, total_clicks):
        #     item = history[i][0]
        #     position = history[i][1]


    # For each item, returns a dictionary of activations.
    def get_activations(self):
        activations = {} # Activation per target per location
        duration_between_clicks = 20.0 # Wait time between two clicks
        session_interval = 50.0 # Wait time between 2 sessions
        session_click_length = 20 # Clicks per session
        total_sessions = math.ceil(self.total_clicks/session_click_length) # Number of sessions so far
        for i in range(0, int(self.total_clicks)):
            session = math.ceil((i+1)/session_click_length) # Session index
            item = self.history[i][0]
            position = self.history[i][1]
            if item not in activations.keys(): activations[item] = {position:0} # Item has not been seen yet. Add to dictionary
            if position not in activations[item].keys(): activations[item][position] = 0 # Item not seen at this position yet. Add to item's dictionary
            time_difference = duration_between_clicks*(self.total_clicks - i) + (total_sessions - session)*session_interval # Difference between time now and time of click
            activations[item][position] += pow(time_difference, -0.5)
        return activations



    # Method to update user state when the time-step is incremented (after taking an adaptation)
    def update(self, menu, number_of_clicks = None):
        # num_clicks = len(self.history)
        #  if len(self.history) < number_of_clicks else number_of_clicks
        # First we add new clicks
        clicks_to_add = self.simple_history()[-number_of_clicks:]
        
        item_list = list(filter(("----").__ne__, menu)) # new menu without separator
        for click in clicks_to_add:
            self.history.append([click, item_list.index(click)])

        # Next we update user expertise
        self.update_freqdist(menu)
        self.activations = self.get_activations()

    # Update frequency distribution based on new history    
    def update_freqdist(self, menu, normalize = True):
        self.freqdist = {}
        for command in menu:
            if command != "----":
                self.freqdist[command] = 0
        
        for item in self.simple_history():
            if item == "": continue
            if item not in list(self.freqdist.keys()):
                self.freqdist[item] = 1.
            else:
                self.freqdist[item] += 1.

        self.total_clicks = sum(list(self.freqdist.values()))

        if normalize: 
            for command in list(self.freqdist.keys()):
                self.freqdist[command] = round(self.freqdist[command]/self.total_clicks, 3)

    # click history without timestamp
    def simple_history(self):
        return [row[0] for row in self.history]

    def __str__(self):
        return str([self.freqdist, self.activations, self.total_clicks])
    def __repr__(self):
        return str([self.freqdist, self.activations, self.total_clicks])
