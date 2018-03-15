import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

########
reviews_train = load_files("train/")
text_train, y_train = reviews_train.data, reviews_train.target
# ------------------------------------------------------------
reviews_test = load_files("test/")
text_test, y_test = reviews_test.data, reviews_test.target
########

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

vect = CountVectorizer(min_df=5).fit(text_train)
vect2r = CountVectorizer().fit(text_test)
X_train = vect.transform(text_train)

X_test = vect.transform(text_test)

feature_names = vect.get_feature_names()

scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

mglearn.tools.visualize_coefficients(
    grid.best_estimator_.named_steps["logisticregression"].coef_, feature_names, n_top_features=40
)
