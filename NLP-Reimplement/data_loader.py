import gensim.downloader as api
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re
from torch.utils.data import Dataset, DataLoader
'''
class TrainData(Dataset):
    """Newsgroup20 Train Data"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
'''

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