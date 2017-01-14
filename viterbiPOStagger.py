#! usr/bin/env python3

#Andrew Johnson
#Computational Linguitics
#December 2016

"""
Part of Speech Tagger using the Viterbi algorithm
on a Hidden Markov Model implemented from scratch, 
with backpointers and backtracking, with and without 
smoothing (Laplace). Using a German training & test 
corpus, over 90% accuracy assigning parts of speech.
"""

###Module 1 - Corpus Reader and Writer###


import collections
from collections import Counter
import math


training_file = 'de-train.tt' #tagged data for training
test_file = 'de-test.t' #untagged data for testing


def process_train(training_file):
    '''Reads a tagged training corpus,
    outputs the observations, tags, and
    tuples of (obs, tag)'''
    training_text = open(training_file, encoding='utf-8')
    training_text = training_text.read()
    
    training_sents = training_text.split('\n\n')
    #Separates by sentences (which have a blank line between them in this data)
    training_sent_init = []
    for sent in training_sents:
        sent = sent.split()
        init_pos = sent[1:2] #Skips the initial wordform and takes the sentence-initial tag
        training_sent_init += init_pos   
    
    training_text = training_text.split()
    #eg: ['Sehr', 'ADV', 'gute', 'ADJ', 'Beratung', 'NOUN']
    training_obs = training_text[::2] #even-numbered entries, eg: ['Sehr', 'gute', 'Beratung',]
    training_tags = training_text[1::2] #odd-numbered entries, eg: ['ADV', 'ADJ', 'NOUN']
    training_tagged = [(training_obs[x], training_tags[x]) for x in range(len(training_obs))]
    #eg: [('Sehr', 'ADV'), ('gute', 'ADJ'), ('Beratung', 'NOUN')]
    
    return training_obs, training_tags, training_tagged, training_sent_init


def process_test(test_file):
    '''Reads an untagged test file, outputs
    the observations broken up by words'''
    test = open(test_file, encoding='utf-8')
    test = test.read()
    test_obs = test.split() #Separates by words
    return test_obs


training_data = process_train(training_file)
training_obs = training_data[0] #even-numbered entries, eg: ['Sehr', 'gute', 'Beratung',]
training_tags = training_data[1]  #odd-numbered entries, eg: ['ADV', 'ADJ', 'NOUN']
training_tagged = training_data[2] #eg: [('Sehr', 'ADV'), ('gute', 'ADJ'), ('Beratung', 'NOUN')]
training_sent_init = training_data[3]

pos = set(training_tags)

test_obs = process_test(test_file)

test_vocab = set(test_obs)
training_vocab = set(training_obs)


###Module 2- Training Procedure###


def find_init_prob(training_sent_init, pos):
    '''From the training data, calculates
    the log of the initial probabilities'''
    count = collections.Counter(training_sent_init) 
    #Gets count of each POS. Equivalent to using FreqDist

    init_probs = {p: count[p]/len(training_sent_init) for p in pos} 
    #Calculates initial probabilities
    #len(training_sent_init) is equivalent to the number of sentences

    #Convert probabilities to log base 2
    for tag in init_probs:
        if init_probs[tag] != 0:
            init_probs[tag] = math.log2(init_probs[tag])
        else:
            init_probs[tag] = float('-inf') #To avoid a log(0) error

    return init_probs


def empty_dic(states):
    dic = {p : 0 for p in pos}
    return dic


def find_trans_prob(training_tags, pos):
    '''From the training data, calculates
    the log of the transition probabilities'''
    total_count = collections.Counter(training_tags)
    #Count of each POS in training data
    
    pos_counts = empty_dic(pos)
    for tag in pos:
        pos_counts[tag]=empty_dic(pos)

    for x in range(len(training_tags)-1):
        pos_counts[training_tags[x]][training_tags[x+1]] += 1 
        #Increments the count for each transition combination encountered

    for tag in pos_counts:
        for word in pos_counts[tag]:
            pos_counts[tag][word] /= total_count[tag] 
            #Divide by total count for each POS to get the probability
            if pos_counts[tag][word] != 0:
                pos_counts[tag][word] = math.log2(pos_counts[tag][word])
                #Convert to log
            else:
                pos_counts[tag][word] = float('-inf')
    
    return pos_counts


def find_emis_prob(training_tagged, pos):
    '''From the training data, calculates
    the log of the emission probabilities'''
    pos_counts = {p : {} for p in pos}

    #Count number of each word for each POS
    for (word, tag) in training_tagged:
        if word not in pos_counts[tag]:
            pos_counts[tag].update({word: 1})
        else:
            pos_counts[tag][word] += 1

    #Count the total number of each POS
    total_count = collections.Counter(pos_counts)
    total_count = {tag:sum(total_count[tag].values()) for tag in pos}

    for tag in pos_counts:
        for word in pos_counts[tag]:
            pos_counts[tag][word] /= total_count[tag]
            #Divide by total count for each emission to get the probability
            if pos_counts[tag][word] != 0:
                pos_counts[tag][word] = math.log2(pos_counts[tag][word]) 
            else:
                pos_counts[tag][word] = float('-inf')
                
    return pos_counts


def laplace_emis_prob(training_tagged, pos): #Laplace version
    '''Calculates the log of the emission probabilities
    and applied Laplace add one smoothing'''
    pos_counts = {p : {} for p in pos}
    V = len(training_vocab) #Number of unique words in training corpus
    
    #Count number of each word for each POS
    for (word, tag) in training_tagged:
        if word not in pos_counts[tag]:
            pos_counts[tag].update({word: 1})
        else:
            pos_counts[tag][word] += 1

    #Count the total number of each POS
    total_count = collections.Counter(pos_counts)
    total_count = {tag:sum(total_count[tag].values()) for tag in pos}

    for tag in pos_counts:
        for word in pos_counts[tag]:
            pos_counts[tag][word] = (pos_counts[tag][word] + 1)/(total_count[tag] + V)
            #(Cw,j + 1) / (C_j + V) to calculate laplace probability,
            #taking away some probability mass from seen items
            if pos_counts[tag][word] != 0:
                pos_counts[tag][word] = math.log2(pos_counts[tag][word]) 
            else:
                pos_counts[tag][word] = float('-inf')

        #The amount of probability mass saved for unseen
        #items for each respective POS
        unseen_values = {tag:0 for tag in pos}
        for tag in unseen_values:
            unseen_values[tag] = 1/(total_count[tag] + V)
                
    return [pos_counts, unseen_values]


init_prob = find_init_prob(training_sent_init, pos)
trans_prob = find_trans_prob(training_tags, pos)
emis_prob = find_emis_prob(training_tagged, pos)

smoothed_emis_prob = laplace_emis_prob(training_tagged, pos)
unseen_values = smoothed_emis_prob[1]
smoothed_emis_prob = smoothed_emis_prob[0]



#Unknown word handling
unknown_words = [w for w in test_vocab if w not in training_vocab]
#Words that were not encountered during training


def add_emis(emis_prob, unknown_words):
    #For words with a probability of 0 for all states, assign a probability of 1 to all states
    for word in unknown_words:
        for j in pos:
            emis_prob[j][word] = 0 
    return emis_prob

emis_prob = add_emis(emis_prob, unknown_words)

def laplace_add_emis(smoothed_emis_prob, unknown_words, unseen_values):
    #For the smoothed emission probabilities
    #For words with a probability of 0 for all states, assign a probability of 1 to all states
    for word in unknown_words:
        for j in pos:
            smoothed_emis_prob[j][word] = unseen_values[j] 
    return smoothed_emis_prob

smoothed_emis_prob = laplace_add_emis(smoothed_emis_prob, unknown_words, unseen_values)


###Module 3- Viterbi Tagging###


def nested_empty_dics(states, T):
    dic = {t : empty_dic_none(states) for t in range(T)}
    return dic


def empty_dic_none(states):  ##Necessary to have again since None instead of 0
    dic = {p : None for p in pos}
    return dic


def viterbi_algorithm(observations, states, init_prob, trans_prob, emis_prob):
    '''Efficiently predicts POS of words in a test set based on a HMM
    using probabilities calculated from an annotated training corpus'''
    N = len(states) #number of states
    T = len(observations) #number of time steps

    viterbi = nested_empty_dics(states, T)    
    backpointer = [empty_dic(states) for t in range(T)]


    #Initialization, aka base case
    t = 0
    for j in states:
        current_obs = observations[t] #string of word at current time stamp
        if current_obs in emis_prob[j]:
            viterbi[t][j] = init_prob[j] + emis_prob[j][current_obs]
        else:
            viterbi[t][j] = float('-inf') 

        backpointer[t][j] = "origin"


    #Recursion step, aka inductive case
    for t in range(1, T):
        current_obs = observations[t]
        for j in states:
            if current_obs in emis_prob[j]:
                options = []
                for i in states:
                    option = (viterbi[t-1][i] + trans_prob[i][j] + emis_prob[j][current_obs], i)
                    options.append(option)
                maximum = max(options) #tuple of (probability, observation string)
                viterbi[t][j] = maximum[0] #max value
                backpointer[t][j] = maximum[1] #POS that had highest value            
            else: #probability of word being in an impossible POS (eg. "cat" as 'DET')
                viterbi[t][j] = float('-inf')


    #Termination step
    last_round = viterbi[T-1]
    #Turns it from dictionary into a list so max can be found
    last_round = [(last_round[key], key) for key in last_round]
    viterbi[T] = max(last_round)[0] #Adds final probability for reference
    predicted_obs = max(last_round)[1]


    #Backtracking step
    final_list = []
    while predicted_obs != "origin":
        final_list += [predicted_obs]
        predicted_obs = backpointer[T-1][predicted_obs]
        T -= 1

    final_list = final_list[::-1] #Reverse order so it goes from first to last
    return final_list


###Module 4- Evaluation###


new_tags = viterbi_algorithm(test_obs, pos, init_prob, trans_prob, emis_prob)

def add_tags(test_file, tags, output_filename):
    '''Adds the predicted tags to the test text,
    formats it according to CoNLL, and saves the file'''
    test = open(test_file, encoding='utf-8')
    test = test.read()
    test = test.split('\n\n') #Sentences are split up by an empty line in this format
    split_test = [sentence.split() for sentence in test]

    tags = iter(tags)
    tagged_test = ''
    for sentence in split_test:
        for word in sentence:
            tagged_test += '%s\t%s\n' % (word, next(tags))
        tagged_test += '\n'
    tagged_test = tagged_test[:-1] #Removes extra '\n' at end

    file = open(output_filename, "w", encoding='utf-8')
    file.write(tagged_test)
    file.close()

    print(output_filename, "has been saved to your directory")

test_file = 'de-test.t'
output1 = "unsmoothed_tagged.tt"
output2 = "smoothed_tagged.tt"

#Viterbi without smoothing
new_tags = viterbi_algorithm(test_obs, pos, init_prob, trans_prob, emis_prob)
add_tags(test_file, new_tags, output1)

#Viterbi with smoothing
new_tags = viterbi_algorithm(test_obs, pos, init_prob, trans_prob, smoothed_emis_prob)
add_tags(test_file, new_tags, output2)

print(" ")
print("In the commant terminal, go to the appropriate directory and enter:")
print("python3 eval.py de-eval.tt", output1)
print("and")
print("python3 eval.py de-eval.tt", output2)

