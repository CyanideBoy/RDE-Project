'''

Taken from https://github.com/Golbstein/pytorch_to_keras/blob/master/pytorch2keras.ipynb

'''

#%matplotlib inline

import torch
from keras import backend as K
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from keras.layers import Conv2D, Input, Dense, MaxPool2D, Flatten, Lambda
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

from model import ConvNet


model = ConvNet(False)
model.cpu()
model.eval()
model.load_state_dict(torch.load('weights/Model_quicksave39.pt'))

print(K.image_data_format())

with tf.device('/cpu:0'):
    inp = Input((400,300,1))
    x = Conv2D(800,(2,300),activation='relu', name='conv1')(inp)
    
    # Reshaping to (BCHW)
    x = MaxPool2D((399,1))(x)
    
    # Reshaping to (BCHW)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)))(x)
    x = Flatten()(x)
    out = Dense(20,name='fc1')(x)
    k_model = Model(inp, out)


trained_weights = model.state_dict()

pytorch_layers = [] # get ptroch layers names and suffixes
for x, l in zip(trained_weights, k_model.layers):
    #print(x)
    #print(l)
    #print(x.find('.'))
    pytorch_layers.append(x[:x.find('.')])

unique_layers = np.unique(pytorch_layers)
print(unique_layers)


for layer in unique_layers:
    weights = trained_weights['{}.weight'.format(layer)].cpu().numpy() # torch weights (nf, ch, x, y)
    biases = trained_weights['{}.bias'.format(layer)].cpu().numpy()
    if 'bn' in layer:
        running_mean = trained_weights['{}.running_mean'.format(layer)].cpu().numpy()
        running_var = trained_weights['{}.running_var'.format(layer)].cpu().numpy()
        W = [weights, biases, running_mean, running_var]
    elif 'fc' in layer:
        biases = trained_weights['{}.bias'.format(layer)].cpu().numpy()
        W = [weights.T, biases]
    else:
        W = [np.moveaxis(weights, [0, 1], [3, 2]), biases] # transpose to (x, y, ch, nf) keras version
    k_model.get_layer(layer).set_weights(W)

k_model.summary()


#### Verify

batch_size = 16

keras_input = np.random.random((batch_size, 400, 300, 1)).astype('float32')
pytorch_input = torch.from_numpy(keras_input.transpose(0,-1,1,2))

p_out = model(pytorch_input.cpu())
k_out = k_model.predict(keras_input)

y = torch.from_numpy(k_out) - p_out
y = y.detach().numpy().flatten()
plt.hist(y);
print('max difference:', y.max(), '\nsum of difffernces:', y.sum())