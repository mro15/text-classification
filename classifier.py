#!/usr/bin/env python3
import argparse
import classifiers
import csv
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
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

def write_results(y_test, y_pred, file_name):
    with open(file_name, 'w') as f:
        result = csv.writer(f)
        result.writerows(zip(y_test, y_pred))

def main():
    opt = read_args()
    global debug
    debug = opt.debug

    x_train, y_train = load_svmlight_file(opt.train)
    x_test, y_test = load_svmlight_file(opt.test)

    #Possible classifiers
    PC = {}
    PC["knn"] = classifiers.Knn
    PC["svm"] = classifiers.Svm
    PC["mlp"] = classifiers.Mlp

    y_preds = []
    y_tests = []
    scores = []
    name = ("-".join(opt.train.split("/")[-1].split("-")[0:2]))
    classifier = PC[opt.classifier](x_train, y_train, x_test, y_test, name=opt.classifier)
    for i in range(0, 10):
        classifier.init()
        classifier.model1_init(name)
        classifier.normalize(preprocessing.MaxAbsScaler())
        classifier.run()
        y_preds.append(classifier.get_y_pred())
        y_tests.append(classifier.y_test)
        scores.append(classifier.get_score())
        write_results(classifier.y_test, classifier.get_y_pred(), "predicts/"+ name + "-" + opt.classifier + str(i))
    best_pos = scores.index(max(scores))
    write_results(y_tests[best_pos], y_preds[best_pos], "predicts/"+ name + "-" + opt.classifier + "-" + "best")

if __name__ == "__main__":
    main()
