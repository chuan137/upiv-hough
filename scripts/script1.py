from numpy import *
import matplotlib.pyplot as plt
import image.tifffile as tif
import image.filter as img_filter
import image.hough as hough
import image.util as util

from skimage.morphology import disk
from skimage.filter.rank import entropy
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage import img_as_float



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f0 = tif.imread('t.tif')
f1 = copy(f0[0, 100:1100, 50:1050])
f1 = img_as_float(f1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bmin = 0.13
bmax = 0.20

def func():
    # crop image
    f1 = copy(f0[100:1100, 50:1050])
    f1[ (f1 < bmin) | (f1 >= bmax) ] = 0


    crops = []
    cropsize = 200
    for i in range(0,1000,cropsize):
        for j in range(0,1000,cropsize):
            crops.append(f1[i:i+cropsize, j:j+cropsize])
    # image crops are stored in crops[]
    # corpsize can be defined


    f2 = copy(crops[18])
    if True:
        f3 = img_filter.sample_importance(f2, 0.9)
    else:
        f3 = copy(f2)




    v, fv = hough.fuzzyHT(f3, 5, 45)

    # maxima?
    r, y0, x0 = where(fv==fv.max())
    r = r[0] + 5
    x0 = x0[0]
    y0 = y0[0]

    # check circle with azimuth
    A, r0, dr, _, h = hough.ring(f2, y0, x0, r)
    dr = int(round(dr))





    # create mask
    f4 = zeros(f2.shape)
    for ir in range(r-dr, r+dr+1):
        rr, cc = draw.circle_perimeter(y0, x0, ir, 'andres')
        f4[rr, cc] = 1
    f5 = zeros(f2.shape)
    f5[where(logical_and(f3, f4))] = 1
    v5, fv5 = hough.fuzzyHT(f5, range(5, 40))


    fv_sub = fv-fv5
    for i, ffv in enumerate(fv_sub): 
        imsave('res_dump/fv-sub-%d.png' % i, ffv)


