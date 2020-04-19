import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from PIL import Image
import matplotlib as mpl
from matplotlib import cm
from matplotlib.cm import plasma, gray

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

NUM_EXAMPLES = 10


# just the classes and matplotlib colormaps for looking at things
classes = ['0','1','2','3','4','5','6','7','8','9']
norm_heat = mpl.colors.Normalize(vmin=0, vmax=1)
cm_plasma = cm.ScalarMappable(norm=norm_heat, cmap=plasma)

# going trough the samples and making some figures
for img_number in range(NUM_EXAMPLES):
	# loading a sample dict and immediately saving the base image
	sample_dict = np.load('SampleDicts/sample_dict_'+str(img_number)+'.npy', allow_pickle=True).item()
	#print(sample_dict['img'][0].shape)
	orig_img = sample_dict['img'][0]
	Image.fromarray((orig_img.numpy()*255).astype(np.uint8)).save('New/Example_{}_base_img.png'.format(img_number))

	# plotting and saving a bar plot of class prediciton probabilities
	plt.figure(figsize=(8,4))
	print(np.exp(sample_dict['model_prediction'][0]))
	plt.bar(np.arange(10), np.exp(sample_dict['model_prediction'][0]))
	plt.xticks(np.arange(10), classes)
	plt.ylim(0,1)
	plt.savefig('New/Example_{}_prediction.png'.format(img_number))
	plt.close()

	
	# saving the reference relevance map computed without any transformation
	normal_relevance = sample_dict['relevance_map_normal'][0,:,:]
	normal_relevance_img = cm_plasma.to_rgba(normal_relevance)
	Image.fromarray((normal_relevance_img*255).astype(np.uint8)).save(
		'New/Example_{}_relevance_reference.png'.format(img_number))

	# initializing the final heatmaps
	# the combined maps are just computed by summing up the relevance maps for the different scales

	# this is where things could definetly be done differently: one could use the maximum of
	# the relvances for the different scales other other things like it
	