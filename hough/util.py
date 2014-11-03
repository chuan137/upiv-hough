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

def neighbour(img, y0, x0, r):
    ymax = img.shape[0]
    xmax = img.shape[1]
    y1 = y0 - r
    x1 = x0 - r
    y2 = y0 + r + 1
    x2 = x0 + r + 1
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0
    if y2 > ymax:
        y2 = ymax
    if x2 > xmax:
        x2 = xmax
    res = np.copy(img[y1:y2, x1:x2])
    return res

def check_pts(y0, x0, l):
    """check if (y0, x0) is in list
        allow +1/-1 error 
    """
    check = []
    for pt in l:
        if abs(pt[0]-y0) <=1 and abs(pt[1]-x0) <=1:
            check.append(True)
        else:
            check.append(False)
    return np.any(check)

