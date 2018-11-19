#!/usr/bin/env python3

import logging
from .baseClassifier import Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from IPython import embed
logging.basicConfig(level=logging.DEBUG)

class Mlp(Classifier):
    def init(self):
        self.score = 0
        self.confusion_matrix = 0
        self.y_pred = 0
        self.model = Sequential()
        self.model.add(Dense(200, activation='relu', input_dim=self.x_train[0].shape[1]))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train, random_state=0)
        self.x_test, self.y_test = shuffle(self.x_test, self.y_test, random_state=0)
        #self.x_train  = self.x_train.todense()
        #self.x_test = self.x_test.todense()
        self.cat_y_train = to_categorical(self.y_train, num_classes=2)
        self.cat_y_test = to_categorical(self.y_test, num_classes=2)

    def run(self):
        self.model.fit(self.x_train, self.cat_y_train, validation_split=0.25, epochs=100, batch_size=128)
        self.score = self.model.evaluate(self.x_test, self.cat_y_test, batch_size=128)
        self.y_pred = self.model.predict_classes(self.x_test)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)

    def get_score(self):
        return self.score

    def get_y_pred(self):
        return self.y_pred.astype(float)

    def get_confusion_matrix(self):
        return self.confusion_matrix
