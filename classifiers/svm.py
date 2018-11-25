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
        if name=="polarity-bow":
            self.c = 0.5
            self.gamma = 0.03125
        elif name=="polarity-w2v":
            self.c = 512
            self.gamma = 0.03125
        elif name=="polarity-tfidf":
            self.c = 2
            self.gamma = 0.5
        elif name=="imdb-w2v":
            self.c = 2
            self.gamma = 0.125
        elif name=="imdb-bow":
            self.c = 32
            self.gamma = 0.0001220703125
        elif name=="mr-bow":
            self.c = 8192
            self.gamma = 0.0078125
        elif name=="mr-tfidf":
            self.c = 8192
            self.gamma = 0.125
        elif name=="mr-w2v":
            self.c = 8192
            self.gamma = 0.0078125
        else: #imdb-tfidf
            self.c = 512
            self.gamma = 0.03125

    def run(self):
        clf = svm.SVC(kernel='linear', C=self.c, gamma=self.gamma, probability=True)
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
