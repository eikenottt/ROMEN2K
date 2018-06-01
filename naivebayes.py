import math

word1 = 1
word2 = 2
treshold = 0.2

sum = 1-(abs(word1 - word2) / (word1 + word2))

if sum > treshold:
    print(1-(abs(word1 - word2) / (word1 + word2)))