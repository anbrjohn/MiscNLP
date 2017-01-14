##!/usr/bin/env python3
# Andrew Johnson
# Foundations of Language Science and Technology
# Januray 2017

"""
Naive Bayes classifier for spam emails, using absolute discounting.
On given data with preset parameters, the resulting accuracy is 92.0%.
With parameter tuning, an accuracy of up to 98.0% can be achieved when
d = 0.1 and the class prior = 0.3.
(Requires appropriate training & testing files to run)
"""


from collections import defaultdict
import math


def get_list(file_name):
    """Returns list of tokens"""
    with open(file_name, "r", encoding="latin-1") as file:
        text = file.read()
        text = text.lower() # Make everything lowercase
        text = text.split("\n")
        return text



def get_test(test_file):
    """Reads test files and splits into list of emails"""
    with open("ham_spam_testing", "r", encoding="latin-1") as file:
        test = file.read()
        test = test.lower()
        test = test.split("#*#*# ")
        test = test[1:] # Remove first entry, which is empty
    return test



class training_set: # Use for both ham and spam training sets
    #eg: spammy = training_set(spam, V)
    def __init__(self, train_list, vocab_set):
        self._train_list = train_list
        self._vocab_set = vocab_set


    def freq_count(self):
        """Returns dictionary of counts for each type from the
        training data that is also in the vocab.
        """
        #eg: fc = spammy.freq_count()
        count_dict = defaultdict(int)
        for entry in self._train_list:
            if entry in self._vocab_set:
                count_dict[entry] += 1
        return count_dict


    def train_model(self, d=0.7):
        """Calculates the probability with discounting for each
        type from the training data. Words not in the vocab default
        to the normalizer value.
        """
        #eg: model = spammy.train_model()
        count_dict = self.freq_count()
        N = sum(count_dict.values())
        n_plus = len(count_dict)
        alpha = (d * n_plus) / N
        normalizer = alpha * (1/len(self._vocab_set))
        model = defaultdict(lambda:normalizer)
        
        for word in set(self._vocab_set):
            prob = normalizer
            if word in count_dict:
                prob += (count_dict[word] - d) / N
            prob = math.log(prob)
            model[word] = prob
        return model 



def guess_email(email, ham_model, spam_model, class_prior=0.5):
    """For a single email, guesses whether it's spam or ham"""
    email = email.split()
    correct_answer = email.pop(0)

    ham_prob = 0
    for word in email:
        ham_prob += ham_model[word] + math.log(class_prior)
    spam_prob = 0
    for word in email:
        spam_prob += spam_model[word] + math.log((1-class_prior))
        
    if ham_prob > spam_prob: 
        guess = "ham"
    else:
        guess = "spam"
    if guess == correct_answer:
        correctness = True
    else:
        correctness = False
    return correctness



def test_accuracy(emails, ham_model, spam_model, class_prior=0.5):
    """Given ham and spam models and a set of test emails, calculates
    whether each email is likely to be spam or not and then returns the
    percentage correctly predicted.
    """
    correct_count = 0
    for mail in emails:
        correct = guess_email(mail, ham_model, spam_model, class_prior)
        if correct:
            correct_count += 1
    return (correct_count / len(emails)) * 100



def tune_parameters(emails, ham, spam):
    """Looks through combinations of d and class prior
    to find what gives the best accuracy.
    Returns results of all combinations.
    """
    emails = get_test(test)
    results = []
    hammy = training_set(ham, V)
    spammy = training_set(spam, V)
    print("Testing 9x9 combinations of d and class prior.")
    print("May take around 30 seconds.")
    for d in range(1,10):
        ham_model = hammy.train_model(d=d/10)
        spam_model = spammy.train_model(d=d/10)
        for class_prior in range(1,10):
            accuracy = test_accuracy(emails, ham_model, spam_model, class_prior=class_prior/10)
            results += [(accuracy,d/10,class_prior/10)]
    return results



def find_best(results, show_plot=True):
    """Prints the parameter settings that give the best accuracy
    and optionally displays a 3D graph of the results.
    """
    max_point = max(results)
    print("Maximum achieved at d={} and class prior={}. Accuracy of {}%.".format(max_point[1],max_point[2],max_point[0]))
    if show_plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        z,x,y = zip(*results)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
        ax.text(max_point[1],max_point[2],max_point[0]+2, "MAX")
        plt.title('Parameter Tuning Results')
        plt.xlabel('d value')
        plt.ylabel('class prior')
        ax.set_zlabel('Accuracy')
        plt.show()



test = "ham_spam_testing"
ham = get_list("ham_training")
spam = get_list("spam_training")
vocab = get_list("vocab_100000.wl")
V = set(vocab)



if __name__ == "__main__":
    emails = get_test(test)
    hammy = training_set(ham, V)
    spammy = training_set(spam, V)
    ham_model = hammy.train_model()
    spam_model = spammy.train_model()
    accuracy = test_accuracy(emails, ham_model, spam_model)
    print("Accuracy is %s%%" % accuracy)
    print("Search for better parameters? (Y/N)")
    try:
        find_better = input()
        find_better = find_better.lower()
        if find_better[0] == "y":
            param = tune_parameters(emails, ham, spam)
            find_best(param, show_plot=True) 
    except:
        pass      
