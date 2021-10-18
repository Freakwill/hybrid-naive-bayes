#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
from typing import List, Union

import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

STOPWORDS = {'，', '。', '：', '；', '、', ',', '.', ' '}

def get_corpus(texts:List[str])->List[str]:
    """cut the words with space in each text
    """
    return [' '.join(w.word for w in pseg.cut(text) if w.word not in STOPWORDS and w.flag not in {'w','t','p','x', 'c', 'uj', 'd', 'un'}) for text in texts]


def get_data(corpus, vectorizer=TfidfVectorizer(stop_words='english')):
    # get corpus by `corpus = get_corpus(texts)`
    vector = vectorizer
    vector.fit(corpus)
    vocabulary = list(vector.vocabulary_.keys())
    vocabulary.sort(key=lambda x:vector.vocabulary_[x])
    return vector.transform(corpus).todense()


def _lda(X, decomposition=LatentDirichletAllocation(n_components=5)):
    # from scipy.special import softmax
    corpus = get_corpus(X)
    X = get_data(corpus)
    decomp = decomposition
    decomp.fit(X.T)
    return np.array([[ci/sum(c)*len(x.split(' ')) for i, ci in enumerate(c)] for c, x in zip(decomp.components_.T, corpus)])

def _tfidf(X):
    corpus = get_corpus(X)
    return get_data(corpus)


from sklearn.svm import SVC
from tpot import TPOTClassifier

class TextClassifierMixin:

    def __init__(self, get_features=_tfidf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_features = get_features

    def fit_text(self, X, y, lazy=False):
        self.Xtrain = X
        self.ytrain = y
        self.n_samples = X.shape[0]
        if not lazy:
            Xtrain = self.get_features(self.Xtrain)
            self.fit(Xtrain, self.ytrain)

    def predict_text(self, X, retrain=True):
        if retrain:
            Xtotal = np.hstack((self.Xtrain, X))
            Xtotal = self.get_features(Xtotal)
            X = Xtotal[self.n_samples:]
            self.fit(Xtotal[:self.n_samples], self.ytrain)
        else:
            X = self.get_features(X)
        return self.predict(X)

    def score_text(self, X, y, retrain=True):
        y_ = self.predict_text(X, retrain)
        return np.mean(y == y_)

from hybrid_bayes import HybridNB
if __name__ == '__main__':
    data = pd.read_csv('../data/data-text.csv', sep='\t')
    X = data['comment']
    y = data['label']
    class TextClassifier(TextClassifierMixin, HybridNB):
        pass
    tc = TextClassifier(binarize=0.0)
    from sklearn.model_selection import *
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    tc.fit_text(X_train, y_train)
    print(tc.score_text(X_test, y_test))

    class TextClassifier(TextClassifierMixin, MultinomialNB):
        pass
    tc = TextClassifier()
    tc.fit_text(X_train, y_train)
    print(tc.score_text(X_test, y_test))

    class TextClassifier(TextClassifierMixin, BernoulliNB):
        pass
    tc = TextClassifier()
    tc.fit_text(X_train, y_train)
    print(tc.score_text(X_test, y_test))

