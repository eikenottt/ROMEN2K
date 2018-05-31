import string
from collections import Counter

punct = str.maketrans(dict.fromkeys(string.punctuation))


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
    word_arr = Counter(
        sentence.translate(punct).lower().split(" "))  # All words to lower-case and split words to Counter
    prob_positive = prob_negative = 1  # Sets sum of positive and negative to 1, to prevent dividing by 0 and accumulate
    # for w in word_arr:  # Loops through words to calculate possibilities
    prob_word_positive = naive_bayes(word_arr, pos)  # Calculates probability of word occurring in positive review
    prob_positive = prob_positive * prob_word_positive  # Multiplies the probabilities together to find probability of whole sentence
    prob_word_negative = naive_bayes(word_arr, neg)  # Repeat same process for negative.
    prob_negative = prob_negative * prob_word_negative
    return prob_positive, prob_negative  # return both probabilities
