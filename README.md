# MiscNLP

##Naive Bayes Classifier##
Naive Bayes classifier for spam emails, using absolute discounting.
On given data with preset parameters, the resulting accuracy is 92.0%.
With parameter tuning, an accuracy of up to 98.0% can be achieved when
discounting factor = 0.1 and the class prior = 0.3.

![My image](https://github.com/anbrjohn/MiscNLP/blob/master/output.png)

##CKY Parser##
Uses the Cocke-Kasami-Younger algorithm for bottom-up
generation of possible parse trees for grammars in
Chomsky normal form 

##Viterbi POS Tagger##
Part of Speech Tagger using the Viterbi algorithm
on a Hidden Markov Model implemented from scratch, 
with backpointers and backtracking, with and without 
smoothing (Laplace). Using a German training & test 
corpus, over 90% accuracy assigning parts of speech.

##Poet Assist##
My first attempt at using NLTK back in 2015.

Scans a text to find 'poetry possibilities'.
For all combinations of words in text, this returns all words
that rhyme (share a rime) with a word in the text while also 
being synonymous with another word in it.
For words with multiple possible pronunciations, this searches through all pronunciations.

eg: synonymous with "sturdy" and rhymes with "route" --> "stout"

synonymous with "shrewd" and rhymes with "route" --> "astute"

Additionally, for words with multiple possible POS,
this applies the NLTK tagger and searches only for words with the same POS.

eg: synonymous with permit (v) and rhymes with "hate" --> "tolerate"

synonymous with permit (n) and rhymes with "hate" --> ∅"""
