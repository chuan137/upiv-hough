from numpy import *
import matplotlib.pyplot as plt
import image.tifffile as tif
import image.filter as ifilter
import image.hough as hough
import image.util as util

from skimage.morphology import disk, square
from skimage.filter.rank import entropy
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage.transform import rescale, hough_circle
from skimage.segmentation import *
from skimage import img_as_float
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage import exposure


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f0 = tif.imread('t.tif')
f1 = copy(f0[0, 100:1100, 50:1050])
f1 = img_as_float(f1)
# f1: convert to float between 0 and 1
# mask: mask of removed rings


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ds_ratio = 1
bsize = 200
rmin = 10
rmax = 30
bmin = 0.12
bmax = 0.20
alpha = 1.0



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# pre-processing
if True:
    f2 = rescale(f1, 1.0/ds_ratio)
    f3 = ifilter.threhold(f2, bmin, bmax)
    #f4 = ifilter.sample_importance(f3, alpha)
# Following steps are included: downsize, threholding filter, 
# and sample importance filter


#votes = hough_circle(f4, arange(rmax+1))
#fvotes = hough.fuzzy(votes)

yy, xx = mgrid[0:1000:bsize, 0:1000:bsize]
for y, x in zip(yy.flatten(), xx.flatten()):
    if 
    img = f4[y:y+bsize, x:x+bsize]
    votes = hough_circle(img, arange(rmax+1))
    fvotes = hough.fuzzy(votes)

    for r in range(rmax, rmin, -1):
        blobs = hough.find_blobs_hessian(fvotes[r])
        print r, blobs[0]



## find blobs
#blobs = []
#for r in range(len(vv)-1,rmin-1,-2):
    #tb, b = hough.find_blobs_hessian(vv[r])
    #for y0, x0, b0 in tb:
        #if hough.test_blob(vv[r-1], y0, x0) or \
            #hough.test_blob(vv[r+1], y0, x0):
                #blobs.append([y0, x0, r, b0])
    #print r

## mask
#mask = zeros(f4.shape)
#for b in blobs:
    #xx, yy = circle_perimeter(b[0], b[1], b[2])
    #mask[xx, yy] = 1

