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
    parser.add_argument('--mode', type=str, help='comparar 10 ou calcular apenas best [all, best]', required=True)
    parser.add_argument('--debug', help='Print debug and load less files', default=False, action="store_true")
    return parser.parse_args()

def pprint(text):
    if debug:
        print(text)

def read_best(file_name):
    y_test = []
    y_pred = []
    with open(file_name, 'r') as f:
        result = csv.reader(f)
        #deve ter um jeito melhor de fazer isso xP
        for line in result:
            y_test.append(line[0])
            y_pred.append(line[-1])
    return y_test, y_pred

def read_all(file_name):
    y_test_all = []
    y_pred_all = []
    for i in range(0, 10):
        y_test = []
        y_pred = []
        with open(file_name+str(i), 'r') as f:
            result = csv.reader(f)
            #deve ter um jeito melhor de fazer isso xP
            for line in result:
                y_test.append(line[0])
                y_pred.append(line[-1])
        y_test_all.append(y_test)
        y_pred_all.append(y_pred)
    return y_test_all, y_pred_all


def main():
    opt = read_args()
    global debug
    debug = opt.debug
    if opt.mode == "best":
        y_test, y_pred = read_best(opt.file)
        print(accuracy_score(y_test, y_pred))
        print(f1_score(y_test, y_pred, average='macro'))
        print(confusion_matrix(y_test, y_pred))
    elif opt.mode == "all":
        y_test, y_pred = read_all(opt.file)
        acc = 0
        f1 = 0
        for i in range(0, 10):
            acc+=accuracy_score(y_test[i], y_pred[i])
            f1+=f1_score(y_test[i], y_pred[i], average='macro')
        print("Score medio: " + str(acc/10))
        print("F1 medio: " + str(f1/10))
    else:
        print("Modo desconhecido")

if __name__ == "__main__":
    main()
