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

class normalize(object):
    
    def __init__(self):
        [self.mean, self.std] = np.load('mean_std.npy')

    def __call__(self, sample):
        return (sample-self.mean)/self.std

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
        
        sample = self.rem_punct(sample)

        return self.matrixify(sample,400)


    def rem_punct(self,sample):
        sample = word_tokenize(sample)
        
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
        
        for r in rem:
            sample.remove(r)

        return sample


    def matrixify(self,sample,lenf):
        matrix = np.zeros((lenf,300))

        for idx,words in enumerate(sample):
            if idx==lenf:
                break
            try:
                matrix[idx] = self.wv[words]
            except KeyError:
                continue
        return matrix

vec_trans = w2v()
#composed_tf = transforms.Compose([vec_trans,normalize(),transforms.ToTensor()])
composed_tf = transforms.Compose([vec_trans, transforms.ToTensor()])


class DatasetFolderWithPaths(datasets.DatasetFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(DatasetFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.samples[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

    def get_words_list(self,index):
        _1, _2, path = self.__getitem__(index)
        text = head_scraper(path)
        proc_text = vec_trans.rem_punct(text)

        return proc_text
    
    def get_custom_matrix(self,index):
          sample, target, path = self.__getitem__(index)
          text = head_scraper(path)
          proc_text = vec_trans.rem_punct(text)
          sample = vec_trans.matrixify(proc_text,len(proc_text))

          return sample

def get_dataset(data_path):

    dset = DatasetFolderWithPaths(root=data_path,
                                        loader = head_scraper,
                                        is_valid_file = cvalid,
                                        transform= composed_tf)

    return dset
