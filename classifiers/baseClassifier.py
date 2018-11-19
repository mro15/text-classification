#!/usr/bin/env python3

class Classifier(object):
    def __init__(self, x_train, y_train, x_test, y_test, name = ""):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.name = name

    def run(self):
        pass

    def init(self):
        pass

    def model1_init(self, name):
        pass

    def normalize(self, scaler):
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

