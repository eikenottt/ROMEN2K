"""
naivebayesclassifier - classifying movie reviews as good or bad

author: John Tore Simosnen
author: Rune Eikemo
version: 1.0
"""

import glob
import random
from collections import Counter
import string
import pickle
import os
import re
import numpy
import math

import time


punct = str.maketrans(dict.fromkeys(string.punctuation))


def make_stopword_list(vocab):
    stopword = []
    for n in vocab:
        word_occ = vocab[n]
        num_of_times_in = (word_occ / len(vocab)) * 100
        if num_of_times_in > 1:
            stopword.append(n)

    for e in stopword:
        vocab.pop(e)

    return stopword


def make_vocab(directory, limit):
    """
    Creates a vocabulary from a directory containing text files
    :param directory:
    :param limit:
    :return:
    """
    i = 0
    arr = Counter()
    for fle in directory:
        if i > limit:
            break
        with open(fle, "r", encoding="utf8") as rtxt:
            rtxt = rtxt.read().replace('\n', '').lower().translate(punct).split(" ")
            arr.update(rtxt)
        i = i + 1
    return arr


def naive_bayes(word, category):
    """
    Calculating the probability of a given word to occur in a category (positive/negative)
    :param word: A word
    :param category: The class positive or negative
    :return: the probably of word occurring in the category
    """
    if category == pos:
        category_values = pos_values
    else:
        category_values = neg_values

    word_occurrence = vocabulary[word]  # the integer of occurrences of the word
    result = 1  # the result value is set to 1 to prevent dividing by zero.
    if word_occurrence != 0:
        word_in_class = category[word] / category_values

        if word_in_class != 0:  # If word doesn't occur in the class
            class_prob = category_values / all_values  # P(c) c = positive || negative
            word_prob = word_occurrence / all_values  # P(w) w = word
            result = word_in_class * (
                    class_prob / word_prob)  # Calculating the probability of the word occurring in class

    return result


def likelihood(sentence):
    """
    Calculates probaility of a sentence/review is positive or negative.

    :param sentence: sentence or reviews to be categorized
    :return: prob_positive, prob_negative: probility of sentence belonging to category positive or negative
    """
    sentence = sentence.translate(punct)  # Removes all the punctuations from given sentence
    word_arr = sentence.lower().split(" ")  # All words to lower-case and split words to array
    frequence_of_words = Counter(word_arr)  # Notes the frequency/occurrence of each word in sentence
    prob_positive = prob_negative = 1  # Sets sum of positive and negative to 1, to prevent dividing by 0 and accumulate
    for w in frequence_of_words.items():  # Loops through words to calculate possibilities
        prob_word_positive = naive_bayes(w[0], pos)  # Calculates probability of word occurring in positive review
        prob_positive = prob_positive * prob_word_positive  # Multiplies the probabilities together to find probability of whole sentence
        prob_word_negative = naive_bayes(w[0], neg)  # Repeat same process for negative.
        prob_negative = prob_negative * prob_word_negative
    return prob_positive, prob_negative  # return both probabilities


def train_model(filename='trained.model', limit=12500):
    """
    Saves a training model in the same folder as the python file is
    If the file already exists, the file is loaded and no training is required

    :return: pos, neg:  The positive and negative vocabulary
    """
    if not os.path.exists(filename):
        neg_files = glob.glob("train/neg/*.txt")
        neg_vocab = make_vocab(neg_files, limit)

        pos_files = glob.glob("train/pos/*.txt")
        pos_vocab = make_vocab(pos_files, limit)

        all_vocab = neg_vocab + pos_vocab

        stopword = make_stopword_list(all_vocab)

        model_dict = {'pos': pos_vocab, 'neg': neg_vocab, 'stopwords': stopword}
        save_file = open(filename, 'wb')
        pickle.dump(model_dict, save_file)
        save_file.close()
    else:
        get_file = open(filename, 'rb')
        model_dict = pickle.load(get_file)
        get_file.close()
        pos_vocab = model_dict['pos']
        neg_vocab = model_dict['neg']
        stopword = model_dict['stopwords']
    return pos_vocab, neg_vocab, stopword


def test_model(test_sentence):
    positive, negative = likelihood(test_sentence)
    print(test_sentence[:20], "...")
    print("Positive sentence: {} \nNegative sentence: {} ".format(positive, negative))
    if positive > negative:
        print("Positive percentage:", (positive / (positive + negative)) * 100)
    else:
        print("Negative percentage:", (negative / (positive + negative)) * 100)
    print()


def test_large_set_of_reviews(directory):
    gsl = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    test_directory = glob.glob(directory)
    random.shuffle(test_directory)
    limit = 30
    test_directory = test_directory[:limit]
    i = 1
    beta = 1
    regex = re.compile('[\\\/](\w+)[\\\/]')
    start_time = time.time()
    for review_path in test_directory:
        class_label = regex.search(review_path).group(1)
        with open(review_path, "r", encoding="utf8") as review:
            key = compare_class_labels(test_single_review(review.read()), class_label)
            review.close()
        gsl[key] += 1
        if i % 10 == 0:
            print(
                "Time spent on {} reviews, {:.2f} seconds:".format(str(i) + "/" + str(limit), time.time() - start_time))
        i += 1
    precision_pos = gsl['tp'] / (gsl['tp'] + gsl['fp'])
    precision_neg = gsl['tn'] / (gsl['tn'] + gsl['fn'])
    recall_pos = gsl['tp'] / (gsl['tp'] + gsl['fn'])
    recall_neg = gsl['tn'] / (gsl['tn'] + gsl['fp'])
    f_mesure = ((beta ** 2 + 1) * precision_pos * recall_pos) / (precision_pos + recall_pos)
    accuracy = (gsl['tp'] + gsl['tn']) / (gsl['tp'] + gsl['fp'] + gsl['tn'] + gsl['fn'])
    error_rate = 1 - f_mesure

    print("TP: {}\nTN: {}\nFP: {}\nFN: {}\n".format(gsl['tp'], gsl['tn'], gsl['fp'], gsl['fn']))
    print("Precision Positive:", precision_pos)
    print("Precision Negative:", precision_neg)
    print("Recall Positive:", recall_pos)
    print("Recall Negative:", recall_neg)
    print("Accuracy:", accuracy)
    print("Error Rate:", error_rate)
    print("F-Measure:", f_mesure)


def test_single_review(review):
    prob_pos, prob_neg = likelihood(review)
    if prob_pos > prob_neg:
        return "pos"
    else:
        return "neg"


def compare_class_labels(label, real_label):
    if label == "pos":
        if label == real_label:
            return 'tp'
        else:
            return 'fp'
    else:
        if label == real_label:
            return 'tn'
        else:
            return 'fn'


if __name__ == '__main__':
    pos, neg, stopword = train_model()
    pos_values = sum(pos.values())  # P(w|c) w = word, c = positive || negative (Conditional probability)
    neg_values = sum(neg.values())  # P(w|c) w = word, c = positive || negative (Conditional probability)
    vocabulary = pos + neg
    all_values = sum(vocabulary.values())  # the integer sum of all words occurring in the vocubalary
    start_time = time.time()
    test_large_set_of_reviews("test/*/*.txt")
    print("Time spent in total {:.2f} seconds".format((time.time() - start_time)))
