#!/usr/bin/env python3

import logging
from .baseClassifier import Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from IPython import embed
logging.basicConfig(level=logging.DEBUG)

class Knn(Classifier):
    def init(self):
        self.score = 0
        self.confusion_matrix = 0
        self.y_pred = 0

    def run(self):
        neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        neigh.fit(self.x_train, self.y_train)
        self.y_pred = neigh.predict(self.x_test)
        self.score = neigh.score(self.x_test, self.y_test)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)

    def get_score(self):
        return self.score

    def get_y_pred(self):
        return self.y_pred

    def get_confusion_matrix(self):
        return self.confusion_matrix
