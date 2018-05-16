"""
naivebayesclassifier - classifying movie reviews as good or bad

author: John Tore Simosnen
author: Rune Eikemo
version: 1.0
"""

import glob
from collections import Counter
import string
from math import log, sqrt

stopword = {"the", "a", "of", "in", "are", "to", "and", "is", "&", "br", "but", "it",
            "that", "as", "was", "for", "on", "be", "width", "have", "its", "one",
            "at", "", "so", "or", "an", "by"}
punct = str.maketrans(dict.fromkeys(string.punctuation))

i = 0

threshold = 0.1


def train_native_bayer(dir, C):
    posdir = glob.glob(dir+C[0]+"*.txt")
    negdir = glob.glob(dir+C[1]+"*.txt")
    ndoc = len(posdir)+len(negdir)


    for c in C:
        dir = glob.glob(dir+c+"*.txt")
        nc = len(dir)
        logprior = nc/ndoc


def make_vocab(dir):
    global arr, i
    for fle in dir:
        if i > 500:
            break
        with open(fle, "r") as rtxt:
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
                diff = differens(dict1[i], dict2[j])
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

    #run naive_bayes function

    return

def naive_bayes(w, c):
    allvocab = pos + neg
    result = 1
    if allvocab[w] != 0:
        allvalues = sum(allvocab.values())
        wordclass = allvocab[w]

        winpos = c[w] / sum(c.values())  # P(w|c) w = word, c = positive || negative

        if winpos != 0:
            poslike = sum(c.values()) / allvalues  # P(c) c = positive || negative
            wordlike = wordclass / allvalues  # P(w) w = word
            result = (winpos) * (poslike / wordlike)

    return result


def likelihood(sentence):
    sentence = sentence.translate(punct)
    sen = sentence.lower().split(" ")
    s = Counter(sen)
    sumpo = sumne = 1
    for w in s.items():
        po = naive_bayes(w[0], pos)
        sumpo *= po
        ne = naive_bayes(w[0], neg)
        sumne *= ne
    return sumpo, sumne




def differens(x, y):
    diff = x - y
    return diff / (x + y)


arr = Counter()

negFiles = glob.glob("train/neg/*.txt")
neg = make_vocab(negFiles)

arr = Counter()
posFiles = glob.glob("train/pos/*.txt")
pos = make_vocab(posFiles)

positive, negative = likelihood("bad,")

print(neg['bad'])

print("Positive sentence: {0:.2f} \nNegative sentence: {1:.2f} ".format(positive, negative))

# compare_dict(pos, neg)

# print("Positive words: {} \nNegative words: {}\n".format(pos, neg))
# print("Stopword list: {}".format(sorted(stopword)))
