import torch
from torchvision import transforms
from torch.utils.data import Dataset

import pandas as pd

import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from gensim import models
import string

a1 = set("-'")
a2 = set(string.ascii_letters + "'-")

def check_set(x):
    if a2.issuperset(x) and not a1.issuperset(x):
        return True
    return False
 

class normalize(object):
    
    def __init__(self):
        [self.mean, self.std] = np.load('mean_std.npy')

    def __call__(self, sample):
        sample['matrix'] =  (sample['matrix']-self.mean)/self.std
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        matrix, class_ = sample['matrix'], sample['class']
        class_ = np.array([class_])
        
        temp = {'matrix': torch.Tensor(matrix),
                'class': torch.Tensor(class_),
                'text': sample['text']}
        return temp

class fasttext(object):
    """
    Convert text to fasttext vectors
    """

    def __init__(self):
        
        self.wv = models.fasttext.load_facebook_vectors('fasttext/crawl-300d-2M-subword.bin')
        nltk.download('punkt')

    def __call__(self, sample):

        sample['text'] = self.rem_punct(sample['text'])
        sample['matrix'] = self.matrixify(sample['text'],80)       ## mean_len = 36, std = 10.02, max = 179
        return sample

    def rem_punct(self,text):
        text = text.replace('&lt;',' ')
        text = text.replace('#36;',' ')
        text = text.replace('\\',' ')
        text = text.strip()
        text = word_tokenize(text)
        
        rem = []
        for w in text:
            if check_set(w):
                continue
            else:
                rem += [w]
            
        for r in rem:
            text.remove(r)

        return text


    def matrixify(self,sample,lenf):
        matrix = np.zeros((lenf,300))

        for idx,words in enumerate(sample):
            if idx==lenf:
                break
            try:
                matrix[idx] = self.wv[words]
            except KeyError:            ## KeyError should not happend in fasttext
                continue
        return matrix

vec_trans = fasttext()
composed_tf = transforms.Compose([vec_trans,normalize(),ToTensor()])
#composed_tf = transforms.Compose([vec_trans, ToTensor()])
#composed_tf = transforms.Compose([vec_trans])


class AGNewsDataset(Dataset):

    def __init__(self, csv_file,transform=None):
    
        self.agnews = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print(idx)

        text = ""
        for sent in self.agnews.iloc[idx, 1:]:
            text = text + sent
        
        sample = {'class': int(self.agnews.iloc[idx, 0])-1, 'text': text}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.agnews)
    
    def get_fixed_data(self,index):
        sample = self.__getitem__(index)
        text = sample['text']        
        matrix = vec_trans.matrixify(text,len(text))
        sample['matrix'] = matrix
        return sample


def get_dataset(data_path):

    dset = AGNewsDataset(csv_file=data_path, transform= composed_tf)

    return dset
