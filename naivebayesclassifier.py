"""
naivebayesclassifier - classifying movie reviews as good or bad

author: John Tore Simosnen
author: Rune Eikemo
version: 1.0
"""

import glob
from collections import Counter
import string
import os
from math import log, sqrt

stopword = {"the", "a", "of", "in", "are", "to", "and", "is", "&", "br", "but", "it",
            "that", "as", "was", "for", "on", "be", "width", "have", "its", "one",
            "at", "", "so", "or", "an", "by"}
punct = str.maketrans(dict.fromkeys(string.punctuation))

i = 0

threshold = 0.1


def train_native_bayer(dir, C):
    posdir = glob.glob(dir + C[0] + "*.txt")
    negdir = glob.glob(dir + C[1] + "*.txt")
    ndoc = len(posdir) + len(negdir)

    for c in C:
        dir = glob.glob(dir + c + "*.txt")
        nc = len(dir)
        logprior = nc / ndoc


def make_vocab(dir):
    global arr, i
    for fle in dir:
        if i > 200:
            break
        with open(fle, "r", encoding="utf8") as rtxt:
            rtxt = rtxt.read().replace('\n', '')

            rtxt = rtxt.lower()

            rtxt = rtxt.translate(punct)

            rtxt = rtxt.split(" ")
            okarray = [s for s in rtxt if s not in stopword]

            temp = Counter(okarray)
            arr = arr + temp
        i = i + 1
    i = 0
    return arr


def compare_dict(dict1, dict2):
    for i in list(dict1):
        for j in list(dict2):
            if i == j:
                diff = difference(dict1[i], dict2[j])
                if diff <= threshold:
                    stopword.add(i)
                    dict1.pop(i)
                    dict2.pop(j)
                    break


def compare_input(pos, neg, input):
    # split input
    # diffarray = []
    # make variables for pos and neg values

    # loop input[i] alias key
    # check if in stopwords
    # pop word
    # continue

    # pos find value on key
    # if value exists
    # save value as possave
    # else
    # value is 0

    # neg find value on key
    # if value exists
    # save value as negsave
    # else
    # value is 0

    # calculate the difference between possave and negsave
    # place difference in array

    # run naive_bayes function

    return


def naive_bayes(word, category):
    """
    Calculating the probability of a given word to occur in a category (positive/negative)
    :param word: A word
    :param category: The class positive or negative
    :return: the probably of word occurring in the category
    """
    vocabulary = pos + neg  # Every word found in both positive and negative dictionary
    result = 1  # the result value is set to 1 to prevent dividing by zero.
    if vocabulary[word] != 0:
        allvalues = sum(vocabulary.values())  # the integer sum of all words occurring in the vocubalary
        word_occurrence = vocabulary[word]  # the integer of occurrences of the word

        word_in_class = category[word] / sum(
            category.values())  # P(w|c) w = word, c = positive || negative (Conditional probability)

        if word_in_class != 0:  # If word doesn't occur in the class
            class_prob = sum(category.values()) / allvalues  # P(c) c = positive || negative
            word_prob = word_occurrence / allvalues  # P(w) w = word
            result = word_in_class * (
                    class_prob / word_prob)  # Calculating the probability of the word occurring in class

    return result


def likelihood(sentence):
    """
    Calculates probaility of a sentence/review is positive or negative.
    :param sentence: sentence or revies to be categorized
    :return: probility of sentence belonging to category positive or negative
    """
    sentence = sentence.translate(punct)  # Removes all the punctuations from given sentence
    word_arr = sentence.lower().split(" ")  # All words to lower-case and split words to array
    frequence_of_words = Counter(word_arr)  # Notes the frequency/occurrence of each word in sentence
    prob_positive = prob_negative = 1  # Sets sum of positive and negative to 1, to prevent dividing by 0 and accumulate
    for w in frequence_of_words.items():  # Loops through words to calculate possibilities
        prob_word_positive = naive_bayes(w[0], pos)  # Calculates probability of word occurring in positive review
        prob_positive *= prob_word_positive  # Multiplies the probabilities together to find probability of whole sentence
        prob_word_negative = naive_bayes(w[0], neg)  # Repeat same process for negative.
        prob_negative *= prob_word_negative
    return prob_positive, prob_negative  # return both probabilities


def difference(x, y):
    diff = x - y
    return diff / (x + y)


arr = Counter()

negFiles = glob.glob("train/neg/*.txt")
neg = make_vocab(negFiles)

arr = Counter()
posFiles = glob.glob("train/pos/*.txt")
pos = make_vocab(posFiles)

test_sentence = 'good good good good '


test_sentence = test_sentence.lower()
test_sentence = test_sentence.translate(punct)

positive, negative = likelihood(test_sentence)

print(neg['bad'])

print("Positive sentence: {} \nNegative sentence: {} ".format(positive, negative))
# dir = ["neg", "pos"]
# print(os.getcwd() + "/train" + "/" + dir[0] + "/*.txt")

# compare_dict(pos, neg)
# print("Positive words: {} \nNegative words: {}\n".format(pos, neg))
# print("Stopword list: {}".format(sorted(stopword)))
