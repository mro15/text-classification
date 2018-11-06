#!/usr/bin/env python3

import gensim
import logging
import numpy as np
from .baseExtractor import Extractor
from IPython import embed
logging.basicConfig(level=logging.DEBUG)

class W2v(Extractor):
    def __tokenize(text):
        ret = []
        for t in text:
            ret.append(gensim.utils.simple_preprocess(str(t)))
        return ret

    def __medianEmbeddingVectorizer(self):
        def transform(w2v, text):
            dim = len(list(w2v.values())[1])
            return [np.mean([self.w2v[t] for t in text if t in self.w2v]
                or [np.zeros(dim)], axis=0)][0]

        train = []
        for i in self.token_train_x:
            train.append(transform(self.w2v, i))

        test = []
        for i in self.token_test_x:
            test.append(transform(self.w2v, i))

        return train, test

    def init(self):
        self.token_train_x = W2v.__tokenize(self.train_data[0])
        self.token_test_x = W2v.__tokenize(self.test_data[0])
        self.workers=10

    def model1_init(self):
        self.size=150
        self.window=2
        self.min_count=1
        self.epochs = 10

    def run(self):
        model = gensim.models.Word2Vec(self.token_train_x, size=self.size,
                window=self.window, min_count=self.min_count, workers=self.workers)
        model.train(self.token_train_x, total_examples=(len(self.token_train_x)),
                epochs=self.epochs)

        self.w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        train,test = self.__medianEmbeddingVectorizer()

        return [train, self.train_data[1]], [test, self.test_data[1]]
