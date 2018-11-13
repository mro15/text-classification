#!/usr/bin/env python3
import argparse
import csv
from IPython import embed
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

debug = False

def read_args():
    parser = argparse.ArgumentParser(description='Os parametros para os extratores são:')
    parser.add_argument('--file', type=str, help='arquivo com as predicoes', required=True)
    parser.add_argument('--debug', help='Print debug and load less files', default=False, action="store_true")
    return parser.parse_args()

def pprint(text):
    if debug:
        print(text)

def read_results(file_name):
    y_test = []
    y_pred = []
    with open(file_name, 'r') as f:
        result = csv.reader(f)
        #deve ter um jeito melhor de fazer isso xP
        for line in result:
            y_test.append(line[0])
            y_pred.append(line[-1])
    return y_test, y_pred

def main():
    opt = read_args()
    global debug
    debug = opt.debug
    y_test, y_pred = read_results(opt.file)
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test, y_pred, average='macro'))
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
