from collections import Counter

testString = ["Good movies are good this one was a good movie", 0]
testString2 = ["Worst movie ever", 1]
testString3 = ["Okay movie, horrible story good acting", 0]
testString4 = ["Give this man an Oscar", 0]
testString5 = ["bad movie", 1]
testString6 = ["Bad habbits actors", 1]
testString7 = ["Movie is best", 0]
testString8 = ["I have never ever watched a movie that sucked this bad", 1]

test = "Bad actors but good movie good bye"

reviews = [testString, testString2, testString3, testString4, testString5, testString6, testString7, testString8]


def sort(arr):
    pos = Counter()
    neg = Counter()
    for r in arr:
        if r[1] == 0:
            pos.update(r[0].lower().split(" "))
        elif r[1] == 1:
            neg.update(r[0].lower().split(" "))

    return pos, neg


pos, neg = sort(reviews)


def naive_bayes(w, c):
    allvocab = pos + neg
    result = 1
    if allvocab[w] != 0:
        allvalues = sum(allvocab.values())
        wordclass = allvocab[w]

        winpos = c[w] / sum(c.values())  # P(w|c) w = word, c = positive || negative
        poslike = sum(c.values()) / allvalues  # P(c) c = positive || negative
        wordlike = wordclass / allvalues  # P(w) w = word

        if winpos != 0:
            result = (winpos) * (poslike / wordlike)

    return result


def likelihood(sentence):
    sen = sentence.lower().split(" ")
    s = Counter(sen)
    sumpo = sumne = 1
    for w in s.items():
        po = naive_bayes(w[0], pos)
        sumpo *= po
        ne = naive_bayes(w[0], neg)
        sumne *= ne
    return sumpo, sumne


positive, negative = likelihood(test)

print("Positive sentence: {0:.2f} %\nNegative sentence: {1:.2f} %".format(positive, negative))

# print("Positive: {}\nNegative: {}".format(pos, neg))
