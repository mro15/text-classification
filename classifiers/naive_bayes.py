#!/usr/bin/env python3

import logging
from .baseClassifier import Classifier
import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from IPython import embed
logging.basicConfig(level=logging.DEBUG)

class NaiveBayes(Classifier):
    def init(self):
        self.score = 0
        self.confusion_matrix = 0
        self.y_pred = 0

    def run(self):
        nb = GaussianNB()
        nb.fit(self.x_train.toarray(), self.y_train)
        self.y_pred = nb.predict(self.x_test.toarray())
        self.score = nb.score(self.x_test.toarray(), self.y_test)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)

    def get_score(self):
        return self.score

    def get_y_pred(self):
        return self.y_pred

    def get_confusion_matrix(self):
        return self.confusion_matrix
