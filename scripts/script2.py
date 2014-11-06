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
mask = zeros(f1.shape)
# f1: convert to float between 0 and 1
# mask: mask of removed rings


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ds_ratio = 2
bmin = 0.12
bmax = 0.20
rmin = 2
rmax = 65
alpha = 1.0


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# detek very bright blob and remove them
if True:
    blobs = blob_doh(f1, threshold=0.002)
    for b in blobs:
        b[2] *= 1
        #mask[b[0]-b[2]:b[0]+b[2]+2, b[1]-b[2]:b[1]+b[2]+2] = disk(b[2])
        #hough.filter_mask()
        f1[b[0]-b[2]:b[0]+b[2]+1, b[1]-b[2]:b[1]+b[2]+1] = f1.mean()


# pre-processing
if True:
    f2 = rescale(f1, 1.0/ds_ratio)
    f3 = ifilter.threhold(f2, bmin, bmax)
    f4 = ifilter.sample_importance(f3, alpha)
#for i in range(10):
#    print i
#    f4 = ifilter.sample_importance(f4, 0.95)
## Following steps are included: downsize, threholding filter, sample importance filter
## The sample importance filter is reapted several times with high 
## alpha to reduce the noise.


# hough transform
if True:
    votes = hough_circle(f4, arange(0, rmax+1))
    fvotes = hough.fuzzy(votes)
vv = copy(fvotes)


# find blobs
mask = zeros(f4.shape)
blobs = []
for i, v in enumerate(vv[-1:5:-1]):
    r = len(vv) - i - 1
    tb, b = hough.find_blobs_hessian(vv[r])
    for _b in tb:
        xx, yy = circle_perimeter(_b[0], _b[1], r)
        mask[xx, yy] = 1
        _b.append(r)
        blobs.append(_b)
    print r



#def fun2(fv):
    #blobs = [ blob_doh(_f) for _f in fv ]
    #blobs = [ item for sublist in blobs for item in sublist ]
    #blobs = array(blobs)

    #if blobs.size == 0:
        #return 0, 0, 0, False, []

    ## find maxima
    #r0, y0, x0 = hough.find_ring(fv)
    #_d = blobs[:,:2] - [y0, x0]
    #_d = [ x[0]**2 + x[1]**2 for x in _d ]
    #_d = [ i for i in _d if i <= 2 ]
    #if len(_d) >= 1:
        #check = True
        #print y0, x0, r0, 'Yes!'
    #else:
        #check = False
        #print y0, x0, r0, 'No!'
    ##fv[r0, y0-1:y0+2, x0-1:x0+2] = 0
    #return r0, y0, x0, check, blobs


#for i in range(1):
    #smin = 30
    #smax = 40
    #r0, y0, x0, dtk0, blobs = fun2(fv[smin:smax])
    #if blobs.size == 0:
        #break
    #if dtk0:
        #msk = hough.gen_ring_mask(fv, f3, y0, x0, smin+r0)
        #_f = hough.filter_mask(f3, msk)
        #_v = hough_circle(_f, arange(0, rmax+1))
        #_fv = hough.fuzzy(_v)

        ## remove hough transform from detected ring
        #fv = fv - _fv
        #fv[fv<0] = 0
    #else:
        #fv[smin+r0-1:smin+r0+2, y0-1:y0+2, x0-1:x0+2] = 0


#def fun1():
    #ss = 20 
    #for _i in range(0,1000,ss):
        #for _j in range(0, 1000, ss):
            #_f = f1[_i:_i+ss, _j:_j+ss]
            #_h = histogram(_f, bins=100)
            #print _h[0].max()
            #pk_half_x = where(_h[0]>0.5*_h[0].max())[0][-1]
            #pk_fourth_x = where(_h[0]>0.10*_h[0].max())[0][-1]
            #pk_half = _h[1][pk_half_x]
            #pk_fourth = _h[1][pk_fourth_x]

            
            ##f1[_i:_i+ss, _j:_j+ss] = ifilter.threhold(_f, pk_half, pk_fourth)
            #_f = ifilter.threhold(_f, pk_half, pk_fourth)
            #_f = ifilter.sample_importance(_f, 0.8)
            #f1[_i:_i+ss, _j:_j+ss] = _f

        ##f1[_i:_i+ss, _j:_j+ss] = exposure.rescale_intensity(_f, in_range=(pk_half, pk_fourth))



        ##pk = where(hist[0]==hist[0].max())[0][0]
        ##pk_half = where(hist[0]>*pk)[0].max()

        ##print hist[1][pk], hist[1][pk_half],
        ##print pk_half, pk_fourth



