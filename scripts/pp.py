from numpy import *
import matplotlib.pyplot as plt
import skimage.color as color
import skimage.draw as draw
from skimage.morphology import disk
from skimage import img_as_float
from skimage.filter.rank import entropy
import image.tifffile as tif
import image.filter as img_filter
import image.hough as hough

rmin = 1
rmax = 31

f4 = copy(f3)

_, fv = hough.fuzzyHT(f3, rmin, rmax)

