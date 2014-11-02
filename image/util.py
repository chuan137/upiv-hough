import matplotlib.pyplot as plt
import numpy as np
import os

def save_imgs(path, name, data):
    if not os.path.exists(path):
        os.makedirs(path)
    for _i, _f in enumerate(data):
        fname = name + '_' + str(_i) + '.png'
        fpath = os.path.join(path, fname)
        plt.imsave(fpath, _f)

def neighbours(f, y0, x0, r):
    g = np.copy(f)
    return g[y0-r:y0+r, x0-r:x0+r]
