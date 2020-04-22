import gensim.downloader as api
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re

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

train_data = ptrain['data']
del ptrain

train_x = np.zeros((len(train_data),400,300))

for idx,data in enumerate(train_data):
    for jdx,words in enumerate(train_data[idx]):
        if jdx==400:
            break
        try:
            train_x[idx,jdx] = wv[train_data[idx][jdx]]
        except KeyError:
            continue

del train_data
del wv
z = train_x.reshape((-1,300))

print(z.shape)
mean = z.mean(axis=0)
std = z.std(axis=0)

np.save('mean_std.npy',[mean,std])