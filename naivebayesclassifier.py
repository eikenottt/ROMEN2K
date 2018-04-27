"""
naivebayesclassifier - classifying movie reviews as good or bad

author: John Tore Simosnen
author: Rune Eikemo
version: 1.0
"""

import glob
from collections import Counter
import string

stopword = {"the", "a", "of", "in", "are", "to", "and", "is", "&", "br", "but", "it",
            "that", "as", "was", "for", "on", "be", "width", "have", "its", "one",
            "at", "", "so", "or", "an", "by"}
punct = string.punctuation
i = 0


def make_vocab(dir):
    global arr, i
    for fle in dir:
        if i > 500:
            break
        with open(fle, "r") as rtxt:
            rtxt = rtxt.read().replace('\n', '')

            rtxt = rtxt.lower()

            for char in punct:
                rtxt = rtxt.replace(char, "")

            rtxt = rtxt.split(" ")
            okarray = [s for s in rtxt if s not in stopword]

            temp = Counter(okarray)
            arr = arr + temp
        i = i + 1
    i = 0
    return arr


arr = Counter()

negFiles = glob.glob("train/neg/*.txt")
neg = make_vocab(negFiles)

arr = Counter()
posFiles = glob.glob("train/pos/*.txt")
pos = make_vocab(posFiles)

print("Positive words: {} \nNegative words: {}".format(pos, neg))
