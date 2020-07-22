import yaml
import os
import torch
from PIL import Image
from torchvision import datasets
import numpy as np
from torch.autograd import Variable as V

    
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def get_model_list(dirname, bv, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    if bv:
        models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                      os.path.isfile(os.path.join(dirname, f)) and "BVM" in f and ".pt" in f]
    else:
        models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                      os.path.isfile(os.path.join(dirname, f)) and "Model" in f and ".pt" in f]
    if models is None:
        return None
    
    models.sort()
    if iteration == 0:
        last_model_name = models[-1]
    else:
        for model_name in models:
            if '{:0>4d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name
    
def save_model(model, optimizer, checkpoint_dir, epoch):
    smodel_name = os.path.join(checkpoint_dir, 'Model_%04d.pt' % epoch)
    opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
    torch.save(model.state_dict(), smodel_name)
    torch.save(optimizer.state_dict(),opt_name)

def save_model_best(model, optimizer, checkpoint_dir, epoch):
    smodel_name = os.path.join(checkpoint_dir, 'BVM_%04d.pt' % epoch)
    opt_name = os.path.join(checkpoint_dir, 'BV_opt.pt')
    torch.save(model.state_dict(), smodel_name)
    torch.save(optimizer.state_dict(),opt_name)

    
def resume(checkpoint_dir, Smodel, optimizer, bv=False):
    last_model_name = get_model_list(checkpoint_dir, bv)
    iteration = int(last_model_name[-7:-3])
    
    Smodel.load_state_dict(torch.load(last_model_name))
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer.pt')))
    
    print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
    
    return iteration+1, Smodel, optimizer





def normalize(R):
    R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
    R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
    return R

def save_maps(data, img_name):
    #n = len(data)
    #print(data.shape)
    data = np.squeeze(data)
    #heatmaps = []
    '''
    for h, heat in enumerate(heatmap):
        print(h,heat.shape)
        maps = hm_to_rgb(heat, scaling=1, cmap = 'seismic')
        heatmaps.append(maps)
        im = Image.fromarray((maps*255).astype(np.uint8))
        im.save('output_heatmap_'+img_name+'.png')
    '''
    for i in range(data.shape[0]):
        maps = normalize(data[i])
        #print(maps.shape)
        im = Image.fromarray((maps*255).astype(np.uint8))
        im.save(img_name[i])
        
        
def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #print('Pred cls : '+str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 1)
    T = (T == np.arange(365)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = V(T)
    return Tt

def fix(paths):
    n = len(paths)
    #print(paths)
    f_paths = [None]*n
    for i in range(n):
        z = paths[i].split('/')
        z[0] = 'rap_maps'
        z = '/'.join(z)
        z = z[:-3] + 'png'
        f_paths[i] = z
    #print(f_paths)
    return f_paths