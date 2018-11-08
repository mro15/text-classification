#!/usr/bin/env python3

import logging
from .baseExtractor import Extractor
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from IPython import embed
logging.basicConfig(level=logging.DEBUG)

class Bow(Extractor):

    def run(self):
        count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,
		                   preprocessor=None, stop_words='english', max_features=None)
        for t in self.train_data[0]:
        	count_vectorizer.fit(t)
        train = []
        for t in self.train_data[0]:
        	train.append(count_vectorizer.transform(t).toarray()[0])
        test = []
        for t in self.test_data[0]:
            test.append(count_vectorizer.transform(t).toarray()[0])
        return [train, self.train_data[1]], [test, self.test_data[1]]
