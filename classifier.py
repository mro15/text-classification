#!/usr/bin/env python3
import argparse
import classifiers
from os import listdir
from os.path import isfile, join
from sklearn.datasets import load_svmlight_file
from IPython import embed

debug = False

def read_args():
    parser = argparse.ArgumentParser(description='Os parametros para os extratores são:')
    parser.add_argument('--train', type=str, help='arquivo de treino', required=True)
    parser.add_argument('--test', type=str, help='arquivo de teste', required=True)
    parser.add_argument('--classifier', type=str, help='Classificar com o seguinte classificador', required=True)
    parser.add_argument('--debug', help='Print debug and load less files', default=False, action="store_true")
    return parser.parse_args()

def pprint(text):
    if debug:
        print(text)

def main():
    opt = read_args()
    global debug
    debug = opt.debug
    
    x_train, y_train = load_svmlight_file(opt.train)
    x_test, y_test = load_svmlight_file(opt.test)

   
    #Possible classifiers
    PC = {}
    PC["knn"] = classifiers.Knn

    classifier = PC[opt.classifier](x_train, y_train, x_test, y_test, name=opt.classifier)
    classifier.init()
    classifier.run()
    print(classifier.get_y_pred())
    print(classifier.get_score())
    print(classifier.get_confusion_matrix())
    
if __name__ == "__main__":
    main()
