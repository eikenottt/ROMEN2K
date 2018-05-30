"""
naivebayesclassifier - classifying movie reviews as good or bad

author: John Tore Simosnen
author: Rune Eikemo
version: 1.0
"""

import glob
from collections import Counter
import string
import pickle
import os
from math import log, sqrt

stopword = {"the", "a", "of", "in", "are", "to", "and", "is", "&", "br", "but", "it",
            "that", "as", "was", "for", "on", "be", "width", "have", "its", "one",
            "at", "", "so", "or", "an", "by"}
punct = str.maketrans(dict.fromkeys(string.punctuation))

threshold = 0.1


def train_native_bayer(dir, C):
    posdir = glob.glob(dir + C[0] + "*.txt")
    negdir = glob.glob(dir + C[1] + "*.txt")
    ndoc = len(posdir) + len(negdir)

    for c in C:
        dir = glob.glob(dir + c + "*.txt")
        nc = len(dir)
        logprior = nc / ndoc


def make_vocab(directory, limit=500):
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
            rtxt = rtxt.read().replace('\n', '')

            rtxt = rtxt.lower()

            rtxt = rtxt.translate(punct)

            rtxt = rtxt.split(" ")
            okarray = [s for s in rtxt if s not in stopword]

            temp = Counter(okarray)
            arr = arr + temp
        i = i + 1
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
        all_values = sum(vocabulary.values())  # the integer sum of all words occurring in the vocubalary
        word_occurrence = vocabulary[word]  # the integer of occurrences of the word

        word_in_class = category[word] / sum(
            category.values())  # P(w|c) w = word, c = positive || negative (Conditional probability)

        if word_in_class != 0:  # If word doesn't occur in the class
            class_prob = sum(category.values()) / all_values  # P(c) c = positive || negative
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
        prob_positive *= prob_word_positive  # Multiplies the probabilities together to find probability of whole sentence
        prob_word_negative = naive_bayes(w[0], neg)  # Repeat same process for negative.
        prob_negative *= prob_word_negative
    return prob_positive, prob_negative  # return both probabilities


def difference(x, y):
    diff = x - y
    return diff / (x + y)


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

        model_dict = {'pos': pos_vocab, 'neg': neg_vocab}
        save_file = open(filename, 'wb')
        pickle.dump(model_dict, save_file)
        save_file.close()
    else:
        get_file = open(filename, 'rb')
        model_dict = pickle.load(get_file)
        get_file.close()
        pos_vocab = model_dict['pos']
        neg_vocab = model_dict['neg']
    return pos_vocab, neg_vocab


def test_model(test_sentence):
    positive, negative = likelihood(test_sentence)
    print(test_sentence[:20], "...")
    print("Positive sentence: {} \nNegative sentence: {} ".format(positive, negative))
    if positive > negative:
        print("Positive percentage:", (positive/(positive+negative))*100)
    else:
        print("Negative percentage:", (negative/(positive+negative))*100)
    print()
    # dir = ["neg", "pos"]
    # print(os.getcwd() + "/train" + "/" + dir[0] + "/*.txt")

    # compare_dict(pos, neg)
    # print("Positive words: {} \nNegative words: {}\n".format(pos, neg))
    # print("Stopword list: {}".format(sorted(stopword)))

# def add_review(sentence, c):
#



if __name__ == '__main__':
    pos, neg = train_model()
    negative_review = "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the " \
                      "terrific sea rescue sequences, of which there are very few I just did not care about any of " \
                      "the characters. Most of us have ghosts in the closet, and Costner's character are realized " \
                      "early on, and then forgotten until much later, by which time I did not care. The character we " \
                      "should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he " \
                      "comes off as kid who thinks he's better than anyone else around him and shows no signs of a " \
                      "cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are " \
                      "well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We " \
                      "are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic " \
                      "here, it was all I could do to keep from turning it off an hour in. "
    positive_review = "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll " \
                      "admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only " \
                      "able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, " \
                      "and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is " \
                      "that it can toy with our emotions. This one did exactly that. The entire theater (which was " \
                      "sold out) was overcome by laughter during the first half of the movie, and were moved to tears " \
                      "during the second half. While exiting the theater I not only saw many women in tears, " \
                      "but many full grown men as well, trying desperately not to let anyone see them crying. This " \
                      "movie was great, and I suggest that you go see it before you judge. "
    test_model(negative_review)
    test_model(positive_review)
