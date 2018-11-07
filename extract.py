#!/usr/bin/env python3
import argparse
import extractors
from os import listdir
from os.path import isfile, join
from sklearn.datasets import dump_svmlight_file
from IPython import embed

debug = False

def read_args():
    parser = argparse.ArgumentParser(description='Os parametros para os extratores são:')
    parser.add_argument('--data', type=str, help='arquivo de data', required=True)
    parser.add_argument('--extractor', type=str, help='Extrair com o seguinte extrator', required=True)
    parser.add_argument('--debug', help='Print debug and load less files', default=False, action="store_true")
    parser.add_argument('--run', type=str, help='Run name', required=False)
    return parser.parse_args()

def pprint(text):
    if debug:
        print(text)

def read_imdb():
    dataset = "datasets/imdb/aclImdb/"

    dado = {}
    dado["test"] = {}
    dado["test"]["pos"] = []
    dado["test"]["neg"] = []
    dado["train"] = {}
    dado["train"]["pos"] = []
    dado["train"]["neg"] = []

    for i in ["test", "train"]:
        for j in ["pos", "neg"]:
            files = listdir(dataset+i+"/"+j+"/")
            for f in files:
                fp = open(dataset+i+"/"+j+"/"+f, 'r')
                dado[i][j].append(fp.readlines())
                if debug and len(dado[i][j]) == 5:
                    break

    train = dado["train"]["pos"] + dado["train"]["neg"]
    label_train = [1]*len(dado["train"]["pos"]) + [0]*len(dado["train"]["neg"])

    test = dado["test"]["pos"] + dado["test"]["neg"]
    label_test = [1]*len(dado["test"]["pos"]) + [0]*len(dado["test"]["neg"])

    return [train, label_train], [test, label_test]

def read_polarity():
    dataset = "datasets/polarity/txt_sentoken/"

    dado = {}
    dado["neg"] = []
    dado["pos"] = []

    for i in ["neg", "pos"]:
        files = listdir(dataset+i+"/")
        for f in files:
            fp = open(dataset+i+"/"+f, 'r')
            dado[i].append(fp.readlines())
            if debug and len(dado[i]) == 10:
                break;

    train_neg = dado["neg"][:int(len(dado["neg"])/2)]
    train_pos = dado["pos"][:int(len(dado["pos"])/2)]
    train = train_pos + train_neg
    label_train = [1]*len(train_pos) + [0]*len(train_neg)

    test_neg = dado["neg"][int(len(dado["neg"])/2):]
    test_pos = dado["pos"][int(len(dado["pos"])/2):]
    test = test_pos + test_neg
    label_test = [1]*len(test_pos) + [0]*len(test_neg)

    return [train, label_train], [test, label_test]

def save_to_file(features, file_out):
    dump_svmlight_file(features[0], features[1], "features/" + file_out)

def main():
    opt = read_args()
    global debug
    debug = opt.debug

    if opt.data == "imdb":
        train,test = read_imdb()
    elif opt.data == "polarity":
        train,test = read_polarity()
    else:
        print("Erro, database nao identificado")
        return 1

    #Possible extractors
    PE = {}
    PE["w2v"] = extractors.W2v
    PE["bow"] = extractors.Bow

    #POGGGG
    if opt.extractor == "w2v":
        extract = PE[opt.extractor](train,test,name=opt.extractor)
        extract.init()
        extract.model1_init()
        train_features, test_features = extract.run()
    elif opt.extractor == "bow":
        extract = PE[opt.extractor](train,test,name=opt.extractor)
        extract.init()
        train_features, test_features = extract.run()
    else:
        print("Erro, extrator nao encontrado")
        return 1
		
    if opt.run:
        run_out = "-" + opt.run
    else:
        run_out = ""
    save_to_file(train_features, opt.data + "-" + opt.extractor + "-train" + run_out)
    save_to_file(test_features, opt.data + "-" + opt.extractor +  "-test" + run_out)

if __name__ == "__main__":
    main()
