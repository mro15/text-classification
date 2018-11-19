#!/usr/bin/env python3

import logging
from .baseClassifier import Classifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from IPython import embed
logging.basicConfig(level=logging.DEBUG)

class Svm(Classifier):
    def init(self):
        self.score = 0
        self.confusion_matrix = 0
        self.y_pred = 0

    def model1_init(self, name):
        pass

    def run(self):
        clf = svm.SVC(kernel='linear', C=32.0, gamma=0.125, probability=True) #TODO: grid search to find better params
        clf.fit(self.x_train, self.y_train)
        self.y_pred = clf.predict(self.x_test)
        self.score = clf.score(self.x_test, self.y_test)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)

    def get_score(self):
        return self.score

    def get_y_pred(self):
        return self.y_pred

    def get_confusion_matrix(self):
        return self.confusion_matrix
