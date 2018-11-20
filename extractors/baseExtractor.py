#!/usr/bin/env python3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import tokenize
from nltk.corpus import subjectivity
import string

class Extractor(object):
    def __init__(self, train_data, test_data, name = ""):
        self.train_data = train_data
        self.test_data = test_data
        self.name = name

    def pre_process(self):
        self.lower_case()
        self.remove_punctuation()
        self.remove_stop_words()
        #self.review_tokenize()
        #self.word_stemming()
        #self.word_lemmatize()
        self.word_to_review()

    def lower_case(self):
        temp_train = []
        temp_test = []
        f = (lambda x:x.lower())
        for t in self.train_data[0]:
            temp_train.append(list(map(f, t)))
        for t in self.test_data[0]:
            temp_test.append(list(map(f, t)))
        self.train_data[0] = temp_train
        self.test_data[0] = temp_test

    def remove_punctuation(self):
        temp_train = []
        temp_test = []
        punctuations = list(string.punctuation)
        f = (lambda x: " ".join(x for x in x.split() if x not in punctuations))
        for t in self.train_data[0]:
            temp_train.append(list(map(f,t )))
        for t in self.test_data[0]:
            temp_test.append(list(map(f, t)))
        self.train_data[0] = temp_train
        self.test_data[0] = temp_test

    def remove_stop_words(self):
        temp_train = []
        temp_test = []
        stop_words = stopwords.words('english')
        f = (lambda x: " ".join(x for x in x.split() if x not in stop_words))
        for t in self.train_data[0]:
            temp_train.append(list(map(f, t)))
        for t in self.test_data[0]:
            temp_test.append(list(map(f, t)))
        self.train_data[0] = temp_train
        self.test_data[0] = temp_test

    def review_tokenize(self):
        temp_train = []
        temp_test = []
        f = (lambda x: word_tokenize(x))
        for t in self.train_data[0]:
            temp_train.append(list(map(f, t)))
        for t in self.test_data[0]:
            temp_test.append(list(map(f, t)))
        self.train_data[0] = temp_train
        self.test_data[0] = temp_test

    def word_stemming(self):
        temp_train = []
        temp_test = []
        s = PorterStemmer()
        f = (lambda x: [s.stem(word) for word in x])
        for t in self.train_data[0]:
            temp_train.append(list(map(f, t)))
        for t in self.test_data[0]:
            temp_test.append(list(map(f, t)))
        self.train_data[0] = temp_train
        self.test_data[0] = temp_test

    def word_lemmatize(self):
        temp_train = []
        temp_test = []
        l = WordNetLemmatizer()
        f = (lambda x: [l.lemmatize(word) for word in x])
        for t in self.train_data[0]:
            temp_train.append(list(map(f, t)))
        for t in self.test_data[0]:
            temp_test.append(list(map(f, t)))
        self.train_data[0] = temp_train
        self.test_data[0] = temp_test

    def word_to_review(self):
        temp_train = []
        temp_test = []
        f = (lambda x: ' '.join(x))
        for t in self.train_data[0]:
            temp_train.append(list(map(f, t)))
        for t in self.test_data[0]:
            temp_test.append(list(map(f, t)))
        self.train_data[0] = temp_train
        self.test_data[0] = temp_test
        print(len(self.train_data[0]))

    def run(self):
        pass

    def init(self):
        pass

    def model1_init(self):
        pass

