import csv
#For fasttext word embedding
# import fasttext
# import fasttext.util
#For word2vec embeddings
# from gensim.models import KeyedVectors
#To compute cosine similarity
from scipy import spatial
import math

# reads a log file and returns a frequency distribution as a dict
def load_click_distribution (menu, filename, normalize = True):
    history = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            history.append(row[0])
    return get_click_distribution(menu, history, normalize)
    

def get_click_distribution(menu, history, normalize = True):
    frequency = {}
    separator = "----"
    for command in menu:
        if command != separator:
            frequency[command] = 0

    item_list = list(filter((separator).__ne__, menu)) #menu without separators
    indexed_history = []
    for item in history:
        indexed_history.append([item, item_list.index(item)])
        if item not in list(frequency.keys()):
            frequency[item] = 1.
        else:
            frequency[item] += 1.
    if normalize:
        total_clicks = sum(list(frequency.values()))
        for command in list(frequency.keys()):
            frequency[command] = round(frequency[command] / total_clicks,3)
    return frequency, total_clicks, indexed_history

# returns frequency distribution given a menu and history
def get_frequencies(menu, history, normalize = True):
    frequency = {}
    total_clicks = len(history)
    menu_items = list(filter(("----").__ne__, menu))
    for command in menu_items:
            frequency[command] = 0
            
    for row in history:
        if row[0] not in list(frequency.keys()):
            frequency[row[0]] = 1.
        else: 
            frequency[row[0]] += 1. 
    
    if normalize:
        for command in list(frequency.keys()):
            frequency[command] = round(frequency[command]/total_clicks, 3)
    
    return frequency, total_clicks

# Computes associatons based on word-embedding models. For each menu item, a list of associated items is returned
def compute_associations(menu):
    # Load pre-trained FT model from wiki corpus
    # ft = fasttext.load_model('../fastText/models/cc.en.300.bin')
    # fasttext.util.reduce_model(ft, 100) 
    # Load pre-trained word2vec models. SO_vectors_200 = software engineering domain
    # model = KeyedVectors.load_word2vec_format('../fastText/models/SO_vectors_200.bin', binary=True)
    model = KeyedVectors.load_word2vec_format('../fastText/models/GoogleNews-vectors-negative300.bin', binary=True)  
    separator = "----"
    associations = {}
    associations_w2v = {}
    for command in menu:
        if command != separator:
            associations[command] = {command:1.0}
            associations_w2v[command] = {command:1.0}

    for i in menu:
        if i == separator: continue
        #Load word vector
        vector1 = ft.get_word_vector(i)
        vector1_word2vec = model.wv[i]
        for j in menu:
            if i == j or j == separator: continue
            vector2 = ft.get_word_vector(j)
            vector2_word2vec = model.wv[j]
            #Compute similarity score
            score = 1 - spatial.distance.cosine(vector1, vector2)
            score_word2vec = 1 - spatial.distance.cosine(vector1_word2vec, vector2_word2vec)
            print(i + "," + j + ": ft = " + str(round(score,3)) + " w2v = " + str(round(score_word2vec,3)) )
            associations[i][j] = score
            associations_w2v[i][j] = score_word2vec
        
    
    # print (associations)
    return associations

    # >>> vector1 = ft.get_word_vector('print')
    # >>> vector2 = ft.get_word_vector('duplicate')

    # >>> 1 - spatial.distance.cosine(vector1,vector2)
    # >>> 1 - spatial.distance.cosine(ft.get_word_vector('asparagus'),ft.get_word_vector('aubergine'))

def load_activations (history):
    total_clicks = len(history)
    activations = {} # Activation per target per location
    duration_between_clicks = 20.0 # Wait time between two clicks
    session_interval = 50.0 # Wait time between 2 sessions
    session_click_length = 40 # Clicks per session
    total_sessions = math.ceil(total_clicks/session_click_length) # Number of sessions so far    
    for i in range(0, int(total_clicks)):
        session = math.ceil((i+1)/session_click_length) # Session index
        item = history[i][0]
        position = history[i][1]
        if item not in activations.keys(): activations[item] = {position:0} # Item has not been seen yet. Add to dictionary
        if position not in activations[item].keys(): activations[item][position] = 0 # Item not seen at this position yet. Add to item's dictionary
        time_difference = duration_between_clicks*(total_clicks - i) + (total_sessions - session)*session_interval # Difference between time now and time of click
        activations[item][position] += pow(time_difference, -0.5)
    return activations

def load_associations (menu, filename):
    separator = "----"
    associations = {}
    for command in menu:
        if command != separator:
            associations[command] = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, skipinitialspace=True)
        for row in csv_reader:
            for item in row:
                if item in associations.keys():
                    associations[item] = associations[item] + row[0:]

    for key in associations:
        if associations[key] == []:
            associations[key] = [key]
    


    # with open(filename) as csv_file:
    #     csv_reader = csv.reader(csv_file)
    #     for row in csv_reader:
    #         if row[0] not in list(associations.keys()):
    #             associations[row[0]] = []
    #         associations[row[0]]= row[1:]
    return associations

def save_menu (menu, filename):
    f = open(filename, "w")
    for command in menu:
        f.write(command + "\n")
    f.close()

def load_menu (filename):
    menu = []
    f = open(filename, "r")
    for line in f:
        line = line.rstrip()
        if len(line) < 2: continue
        menu.append(line)
    return menu

# Returns association matrix for a menu using the associations dictionary
def get_association_matrix(menu, associations):
    association_matrix = []
    for k in range (0, len(menu)):
        if menu[k] in associations:
            for l in range (0, len(menu)):
                if menu[l] in associations[menu[k]]:
                    association_matrix.append(1.0)
                else:
                    association_matrix.append(0.0)
        else:
            for l in range (0, len(menu)):
                association_matrix.append(0.0)
    return association_matrix

# Returns sorted frequencies list for a menu using the frequency dictionary
def get_sorted_frequencies(menu,frequency):
    separator = "----"
    sorted_frequencies = []
    for k in range (0, len(menu)):
        if menu[k] == separator:
            sorted_frequencies.append(0.0)
        else:
            sorted_frequencies.append(frequency[menu[k]])
    return sorted_frequencies

    
def get_assoc_and_freq_list(state):
    separator = "----"
    associations = state.menu_state.associations
    frequency = state.user_state.freqdist
    menu = state.menu_state.menu
    # total_clicks = state.user_state.total_clicks
    # associations = load_associations(menu, filename)
    # frequency, total_clicks = load_click_distribution(menu, filename)
    assoc_list = []
    freq_list = []

    for k in range(0, len(menu)):
        if menu[k] in associations:
            for l in range(0, len(menu)):
                if menu[l] in associations[menu[k]]:
                    assoc_list.append(1.0)
                else:
                    assoc_list.append(0.0)
        else:
            for l in range (0, len(menu)):
                assoc_list.append(0.0)
    
    for k in range(0, len(menu)):
        if menu[k] == separator:
            freq_list.append(0.0)
        else:
            freq_list.append(frequency[menu[k]])
    return assoc_list, freq_list

def get_header_indexes(menu):
        header_indexes = []
        separator = "----"
        groupboundary = False
        for i in range(0, len(menu)):
            if i == 0 or menu[i] == separator:
                groupboundary = True # Found a group start indicator
            if groupboundary and menu[i] != separator:
                header_indexes += [i] # First item of group (header)
                groupboundary = False
        return header_indexes