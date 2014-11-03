from numpy import *
import matplotlib.pyplot as plt
import hough.tifffile as tif
import hough.filter as ifilter
import hough.hough as hough
import hough.util as util

from skimage.morphology import disk
from skimage.filter.rank import entropy
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage.transform import rescale, hough_circle
from skimage.segmentation import *
from skimage import img_as_float
from skimage.feature import blob_dog, blob_log, blob_doh


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f0 = tif.imread('t.tif')
f1 = copy(f0[0, 100:1100, 50:1050])
f1 = img_as_float(f1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bmin = 0.12
bmax = 0.20
rmin = 40
rmax = 50
alpha = 0.5


# detek blob
blobs = blob_doh(f1, threshold=0.002)
for b in blobs:
    b[2] *= 1.2
    f1[b[0]-b[2]:b[0]+b[2]+1, b[1]-b[2]:b[1]+b[2]+1] = f1.mean()

# downsize
# threholding filter
f2 = rescale(f1, 0.5)
f3 = ifilter.threhold(f2, bmin, bmax)
f4 = ifilter.sample_importance(f3, alpha)

# hough transform
v = hough_circle(f4, arange(rmin, rmax))
fv = hough.fuzzy(v)

for i in range(10):
    # find candidate
    r0, y0, x0 = hough.find_ring(fv)
    print y0, x0, r0,

    # check blob
    c0 = util.neighbour(fv[r0-1], y0, x0, 20)
    c1 = util.neighbour(fv[r0], y0, x0, 20)
    c2 = util.neighbour(fv[r0+1], y0, x0, 20)
    b1 = blob_doh(c1, max_sigma=10, num_sigma=20)

    if b1.size == 0:
        print b1
        fv[r0, y0-r0:y0+r0+1, x0-r0:x0+r0+1] *= 0.5
        continue
    else:
        chk1 = b1 - [20, 20, 0]
        check = chk1[:,0]**2 + chk1[:,1]**2
    
    # rm ring
    # m is the mask of the found ring
    # c is the coefficients fitted to the gauss function
    # _fv is the fuzzied votes from mask
    # which is then removed from fv
    if check.min() <= 2:
        m, c = hough.mask_ring(fv, f4, y0, x0, r0, rmin)
        _v = hough_circle(m, arange(rmin, rmax))
        _fv = hough.fuzzy(_v)
        fv = fv - _fv
    else:
        _y, _x, _r =  chk1[check==check.min()][0]
        _y += y0; _x += x0
        fv[r0, _y-_r:_y+_r+1, _x-_r:_x+_r+1] *= 0.5
        print _y, _x, _r


    # print the found ring
    if check.min() <= 2:
        print 'YES!'
    





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


