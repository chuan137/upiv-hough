import numpy
import random


def pixel_neighbour_mean(img, i, j, r):
    """return pixel value mean in center and neighbour of range (r)"""
    psum = 0
    norm = 0
    sx, sy = img.shape
    for ki in range(i-r, i+r+1):
        for kj in range(j-r, j+r+1):
            if ki>0 and ki<sx and kj>0 and kj<sy:
                psum += img[ki, kj]
                norm += 1
    return psum/norm


def sample_importance(img, alpha, r=1, rn=1):
    """ return binary image, importance sampled """
    m  = img[img>0].mean()
    f0 = numpy.copy(img)
    for i, j in numpy.vstack(numpy.where(img>0)).T:
        ran = random.uniform(0,1) 
        # .. higher alpha means more pixel remained 
        # .. maximum alpha is 1, when all pixel kept
        # .. w is the mean of neighbouring pixels, normalized by 
        #       mean value of all nonzero pixels of [img]
        #w = pixel_neighbour_mean(img,i,j,r)/m
        w = pixel_neighbour_mean(img,i,j,r)/pixel_neighbour_mean(img,i,j,r+rn)
        if w > 1:
            w = w * 1.2
        else:
            w = 0.8 * w 
        if ran*w > 1 - alpha:
            f0[i,j] = 1
        else:
            f0[i,j] = 0
    return f0     

def threhold_high(img, alpha):
    res = numpy.copy(img)
    res[res<alpha*img.max()] = 0
    return res

def threhold(img, alpha, beta, mode='abs'):
    res = numpy.copy(img)
    if mode == 'abs':
        res[(res<alpha) | (res>beta)] = 0
    else:
        res[(res<alpha*img.max()) | (res>beta*img.max())] = 0
    return res
