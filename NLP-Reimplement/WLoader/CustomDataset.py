import torch
from torchvision import transforms, datasets

import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re
import os 
from gensim import models

def cvalid(file):

  ext = os.path.splitext(file)[-1]
  if ext=='':
    return True
  
  return False

def head_scraper(file):
    
    with open(file, mode='r', encoding='utf8', errors='ignore') as f:
        text = f.read()
        proc_text = '\n'.join((text.split('\n\n'))[1:])
    
    return proc_text



class w2v(object):
    """
    Convert text to w2v vectors
    """

    def __init__(self):
        # Load w2v?
        self.wv = models.KeyedVectors.load_word2vec_format(
                            'GoogleNews-vectors-negative300.bin', binary=True)

        nltk.download('punkt')

        

    def __call__(self, sample):
        
        sample = word_tokenize(sample)
        #print(sample)
        rem = []
        for w in sample:
            if w=='.':
                continue
            if w=="'":
                continue
            if re.search('[a-zA-Z]', w):
                continue
            else:
                rem += [w]
        #print(rem)
        for r in rem:
            sample.remove(r)

        matrix = np.zeros((400,300))

        for idx,words in enumerate(sample):
            if idx==400:
                break
            try:
                matrix[idx] = self.wv[words]
            except KeyError:
                continue
        return matrix

composed_tf = transforms.Compose([w2v(),transforms.ToTensor()])

def get_dataset(data_path):

    extensioN = ['']
    dset = datasets.DatasetFolder(root=data_path,
                                        loader = head_scraper,
                                        is_valid_file = cvalid,
                                        transform= composed_tf)

    return dset
