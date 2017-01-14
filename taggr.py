#!/usr/bin/python3
# Andrew Johnson
# December 2016

'''
Uses the Cocke-Kasami-Younger algorithm for bottom-up
generation of possible parse trees for grammars in
Chomsky normal form 
'''

import nltk
from nltk.tree import *
from nltk.draw.tree import draw_trees


def makegram(grammar):
    ''' Converts grammar from file to dictionary.

        Args:
            Grammar file in Chomsky Normal Form (.cfg or .txt)

        Returns:
            Stores rules in dictionary in which the key is the *output* of
            the rule and the value is the input.
                eg: ["book"]:["NN", "VB"]
                    [NP]:["DET NN"]
    '''
    grammar = open(grammar, encoding='utf-8')
    grammar = grammar.read()

    grammar = grammar.split('\n') # Splits into list of rules
    del(grammar[-1]) # File ends in blank line. Don't need.
    grammar = [rule.split(' -> ') for rule in grammar] # Splits rules into [lefthand string, righthand string]

    gram_dict = {}
    for rule in grammar:
        left = rule[0] # Lefthand side of rule
        right = rule[1].strip('"') # Righthand side of rule, with quotes removed around terminal symbols
        if right not in gram_dict:
            gram_dict[right] =[left]
        else:
            gram_dict[right] += [left]
    return gram_dict


def makechart(length):
    '''Initializes a CKY table, with keys as tuples (i,k)
       and all values as empty lists. Note: Starts at 1, not 0.

       Args:
           Length of chart (Number of tokens in sentence)

       Returns:
           Dictionary with values as empty lists
           eg: (1,2):[]
    '''
    chart = {}
    for i in range(1, length+1): # i represents the horizontal rows
        for k in range(i+1, length+2): # k represents the vertical columns
            chart[(i,k)] = []
    return chart


def cky(words, grammar):
    '''Fills in the CKY chart bottom-up.

       Args:
           words: A string of words
           grammar: Grammar provided from makegram() function

       Returns:
           chart: Chart from makechart() function filled out with
              backpointers. If a word not in the grammar is encountered,
              returns the mostly/completely empty chart.

    '''
    words = words.split()
    length = len(words)
    chart = makechart(length)
    grammar = makegram(grammar)

    # Outermost layer, nonterminal symbols only
    for i in range(1, length+1):
        word = words[i-1]

        #Check for words not in grammar
        if word in grammar:
            rules = grammar[word]
        else:
            print("\tThe word '%s' is not in grammar!" % (word))
            return chart

        rules = [[rule, word] for rule in rules] # Adds wordform as second element
        # Add to chart. This functions as the backpointer termination condition.
        chart[(i, i+1)] += rules

    #Moving through higher layers of chart
    for b in range(2, length+1): # b represents the width
        for i in range(1, length-b+2):
            for k in range(1, b):
                rulesB = chart[(i, i+k)]
                rulesC = chart[(i+k, i+b)]

                for B in rulesB: # There could be more than one rule for B or C (or none)
                    for C in rulesC:
                        justB = B[0] # To exclude backpointer element
                        justC = C[0]
                        BC = justB+" "+justC
                        if BC in grammar:
                            A = grammar[BC]
                            # Adds backpointer value, equal to i+k
                            A = [[rule, (i+k,justB,justC)] for rule in A]
                            chart[(i,i+b)] += A
    return chart


def backpointer(i,k,A,chart):
    '''Rewinds through CKY chart to recursively generate all possible parse trees.

       Args:
           i: Cell position along horizontal axis. Starts at 1.
           k: Cell position along vertical axis. Starts at length of words + 1.
           A: The lefthand rule you are looking for. For full sentences, starts as "SIGMA".
           chart: CKY chart generated in cky function

       Returns:
           trees: List of all parse trees in nltk.tree format
    '''
    parses= chart[(i,k)] # Grabs possibilities for given location
    options = set([(parse[0],parse[1]) for parse in parses]) # Remove doubles
    options = [choice for choice in options if choice[0] == A] # Removes parses with wrong parent
    trees = []

    for parse in options: #Iterate over possible tree parents
        if type(parse[1]) != str: # For nonterminal branches
            A,(j,B,C) = parse #Instantiate necessary variables at once

            #Recurse over both respective child nodes
            #Returned as list of possible sub-trees
            B_trees = backpointer(i,j,B,chart)
            C_trees = backpointer(j,k,C,chart)

            #For all possible combinations of sub-trees
            for b in B_trees:
                for c in C_trees:
                    tree = Tree(A,[b,c])
                    trees += [tree]

        else: # For terminal branches
            terminal_tree = Tree(A,[parse[1]])
            trees += [terminal_tree]

    return trees


def recognizer(text):
    """Recognizes whether a string contitutes a sentence.

       Args:
           text: String to be tested

       Returns: Boolean True or False
    """
    length = len(text.split())
    chart = cky(text, gram) #Makes new chart based on the input
    trees = backpointer(1, length+1, "SIGMA", chart)
    if len(trees) > 0:
        return True
    else:
        return False


def number_of_parses(input_filename, output_filename):
    """Given a text, calculates the number of parses for each
       sentence and saves a text file with the original sentences
       and the number of parses for each.
    """
    sentences = open(input_filename, encoding='utf-8')
    sentences = sentences.read()
    sentences = sentences.split('\n') # Splits each line
    del(sentences[-1]) # Raw text ends in blank line.
    sentences = sentences[:15]

    number_of_sentences = len(sentences)
    current_s = 1
    new_text = '' #This will eventually be saved as the new file

    for sent in sentences:
        #Display an update
        print("Processing sentence %d out of %d..." % (current_s, number_of_sentences))
        length = len(sent.split())
        chart = cky(sent, gram) #Make new chart for each sentence
        trees = backpointer(1, length+1, "SIGMA", chart) #Rewinds through the backpointers to list trees

        new_text += sent
        new_text += "\t%d\n" % (len(trees))

    print("\nComplete")

    new_file = open(output_filename, "w", encoding='utf-8')
    new_file.write(new_text)
    new_file.close()

    print(output_filename, "has been saved to your directory.")


def draw_nice(trees_to_draw, maxdraw=5):
    """Produces all posisble syntax trees (preset
       to only go up to 5). Shown on screen and can be
       saved as postscript files.
    """
    length = len(trees_to_draw.split())
    chart = cky(trees_to_draw, gram)
    trees = backpointer(1,length+1,"SIGMA",chart)
    if len(trees) > maxdraw:
        print("Warning, large number of parses.")
        print("Only showing first %d trees." % (maxdraw))
    for t in trees[:maxdraw]:
        draw_trees(t)


gram = 'atis-grammar-cnf.cfg'
input_filename = "atis-test-sentences.txt"
output_filename = "number_of_trees.txt"
trees_to_draw = "can you tell me about the flights from saint petersburg to toronto again ."
