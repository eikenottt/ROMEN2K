import glob
import string
from collections import Counter

punct = str.maketrans(dict.fromkeys(string.punctuation))

stopword = []


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
            rtxt = rtxt.read().replace('\n', '')

            rtxt = rtxt.lower()

            rtxt = rtxt.translate(punct)

            rtxt = rtxt.split(" ")
            okarray = [word for word in rtxt if word not in stopword]

            temp = Counter(okarray)
            arr = arr + temp
        i = i + 1
    return arr


if __name__ == "__main__":
    file = glob.glob("train/neg/*.txt")
    neg = make_vocab(file, 1000)
    sum_of_all_words = sum(neg.values())
    for n in neg:
        word_occ = neg[n]
        num_of_times_in = (word_occ / len(neg)) * 100
        if num_of_times_in > 0.5:
            stopword.append(n)

    print(len(neg))
    for e in stopword:
        neg.pop(e)

    print(stopword)
    print(len(neg))
