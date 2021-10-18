#!/usr/bin/env python


from sklearn.naive_bayes import *
from sklearn.preprocessing import binarize
from sklearn.utils.validation import _deprecate_positional_args


class HybridNB(MultinomialNB):
    """Hybrid Naive Bayes classifier, extends MultinomialNB

    The hybrid model estimates the probability in the training phase by Bernoulli Model.
    The training datasets are normally adequate; without the term frequency information,
    this model should work fine during training.
    In the predicting phase, word frequency plays an important role.
    Multinomial Model is applied to estimate the probability. 

    *Reference*
    R. Chiong and Lau Bee Theng, "A hybrid Naive Bayes approach for information filtering," 
    2008 3rd IEEE Conference on Industrial Electronics and Applications, 2008:1003-1007.
    """
    
    @_deprecate_positional_args
    def __init__(self, *, binarize=0.0, **kwargs):
        super().__init__(**kwargs)
        self.binarize = binarize

    def _check_X_y(self, X, y):
        X, y = super()._check_X_y(X, y)
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return X, y
