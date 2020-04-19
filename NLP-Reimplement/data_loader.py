import gensim.downloader as api
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re

def train_data():

    wv = api.load('word2vec-google-news-300')

    ptrain = fetch_20newsgroups(subset='train',shuffle='True')
    ptrain['data'] = ['\n'.join((x.split('\n\n'))[1:]) for x in ptrain['data']]

    nltk.download('punkt')

    ptrain['data'] = [word_tokenize(x) for x in ptrain['data']]

    for idx,y in enumerate(ptrain['data']):
        rem = []
        for w in y:
            if w=='.':
                continue
            if w=="'":
                continue
            if re.search('[a-zA-Z]', w):
                continue
            else:
                rem += [w]
        
        for r in rem:
            y.remove(r)

        ptrain['data'][idx] = y

    train_data, train_y = ptrain['data'], ptrain['target']

    train_x = np.zeros((len(train_data),400,300))
    
    for idx,data in enumerate(train_data):
        for jdx,words in enumerate(train_data[idx]):
            if jdx==400:
                break
            try:
                train_x[idx,jdx] = wv[train_data[idx][jdx]]
            except KeyError:
                continue

    return train_x, train_y


def test_data():

    wv = api.load('word2vec-google-news-300')

    ptest = fetch_20newsgroups(subset='test',shuffle='True')
    ptest['data'] = ['\n'.join((x.split('\n\n'))[1:]) for x in ptest['data']]

    nltk.download('punkt')

    ptest['data'] = [word_tokenize(x) for x in ptest['data']]

    for idx,y in enumerate(ptest['data']):
        rem = []
        for w in y:
            if w=='.':
                continue
            if w=="'":
                continue
            if re.search('[a-zA-Z]', w):
                continue
            else:
                rem += [w]
        
        for r in rem:
            y.remove(r)

        ptest['data'][idx] = y

    test_data, test_y = ptest['data'], ptest['target']

    test_x = np.zeros((len(test_data),400,300))
    
    for idx,data in enumerate(test_data):
        for jdx,words in enumerate(test_data[idx]):
            if jdx==400:
                break
            try:
                test_x[idx,jdx] = wv[test_data[idx][jdx]]
            except KeyError:
                continue

    return test_x, test_y