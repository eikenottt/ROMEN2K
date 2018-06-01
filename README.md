# Naive Bayes Classifier
This classifier is trained on 25.000 IMDB movie reviews to determine whether a given movie
review is considered positive or negative.

## Folder Hierarchy
The folder hierarchy should look like this:

```
Project folder
    |- train
    |  |- neg
    |  |- pos
    |- test
    |   |- neg
    |   |- pos
    |- naivebayesclassifier.py
    ...
```
**Notice:**
*The project must contain the test folder as shown. If you already have a trained model, the train folder is optional.*
You should add your own movie review dataset if you want to retrain the model in the fashion shown above.

## Training the classifier
If the training model is not located in the same folder as the python script, the training will take place.
This function looks through every positive and negative movie review in the training folder by default.
This can be changed in the parameter of the `train_model()` function by defining a limit from 0 - 12.500.

## Testing a movie review
To predict the sentiment of a single review, use the `test_model(your_review)` function. If you want to check a
larger set of reviews use the `test_large_set_of_reviews(directory)` function.

