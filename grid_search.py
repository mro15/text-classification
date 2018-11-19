#!/usr/bin/env python3
import argparse
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from IPython import embed
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy

debug = False

def read_args():
    parser = argparse.ArgumentParser(description='Os parametros para os extratores são:')
    parser.add_argument('--train', type=str, help='arquivo de treino', required=True)
    parser.add_argument('--debug', help='Print debug and load less files', default=False, action="store_true")
    return parser.parse_args()

def pprint(text):
    if debug:
        print(text)

def write_results(file_name, result):
    with open(file_name, 'w') as f:
        f.write(str(result))

def GridSearch(X_train, y_train):

	# define range dos parametros
	C_range = 2. ** numpy.arange(-5,15,2)
	gamma_range = 2. ** numpy.arange(3,-15,-2)
	k = [ 'rbf']
	#k = ['linear', 'rbf']
	param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

	# instancia o classificador, gerando probabilidades
	srv = svm.SVC(probability=True)

	# faz a busca
	grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=True)
	grid.fit (X_train, y_train)

	# recupera o melhor modelo
	model = grid.best_estimator_

	# imprime os parametros desse modelo
	return grid.best_params_


def main():
    opt = read_args()
    global debug
    debug = opt.debug

    x_train, y_train = load_svmlight_file(opt.train)
    best = GridSearch(x_train, y_train)

    name = ("-".join(opt.train.split("/")[-1].split("-")[0:2]))
    write_results("best_params/"+ name, best)

if __name__ == "__main__":
    main()
