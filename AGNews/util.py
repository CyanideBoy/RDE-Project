from gensim import models
import numpy as np
from matplotlib import cm, transforms
from matplotlib import pyplot as plt
from fasttext import CustomDataset

def normalize():

    x = CustomDataset.get_dataset('data/train.csv')
    train_x = np.zeros((len(x),80,300))
    
    for g in range(len(x)):
        train_x[g] = x[g]['matrix'].numpy()

    train_x = train_x.reshape(-1,300)
    
    print(train_x.shape)
    
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)

    np.save('mean_std.npy',[mean,std])

    return [mean,std]


def plot_text_heatmap(words, scores, counter, width=14, height=0.3, verbose=0, max_word_per_line=15):
    fig = plt.figure(figsize=(width, height))
    
    ax = plt.gca()

    ax.set_title(str(counter), loc='left')
    tokens = words
    if verbose > 0:
        print('len words : %d | len scores : %d' % (len(words), len(scores)))

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)
    
    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]

    # ^^^^ For our application 0 to 1 is enough but it gives rather poor representation as most words mapped to 0 would be
    # In blue!

    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    
    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)

    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(0.0, loc_y, token, size='large',
                       bbox={#'width':12,
                             #'height':2,
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'square,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  2.0
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+19, units='dots')

    if verbose == 0:
        ax.axis('off')

    fig.savefig('heatmap'+str(counter)+'.png')