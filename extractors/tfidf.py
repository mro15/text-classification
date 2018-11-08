#!/usr/bin/env python3

import logging
from .baseExtractor import Extractor
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython import embed
logging.basicConfig(level=logging.DEBUG)

class Tfidf(Extractor):

    def run(self):
        vectorizer = TfidfVectorizer()
        for t in self.train_data[0]:
        	vectorizer.fit(t)
        train = []
        for t in self.train_data[0]:
            train.append(vectorizer.transform(t).toarray()[0])
        test = []
        for t in self.test_data[0]:
            test.append(vectorizer.transform(t).toarray()[0])
        return [train, self.train_data[1]], [test, self.test_data[1]]
