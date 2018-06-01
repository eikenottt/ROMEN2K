# Naive Bayes Classifier
This classifier is trained on 25.000 IMDB movie reviews to determine whether a given movie
review is considered positive or negative.

## Folder Hierarchy

```Project folder
    |_ train
       |_neg
       |_pos
    |_ test
        |_neg
        |_pos
    naivebayesclassifier.py
```

## Training the classifier
If the training model is not located in the same folder as the python script, the training will take place.
This function looks through every positive and negative movie review in the training folder by default.
This can be changed in the parameter of the `train_model()` function by defining a limit from 0 - 12.500.

## Testing a movie review
To test the