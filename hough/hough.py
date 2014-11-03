import numpy as np

def HT(img, rho):
    shape = img.shape
    vote = np.zeros((len(rho), shape[0], shape[1]))
    def count(vote, ri, a, b):
        try:
            vote[ri, a, b] += 1
        except:
            pass
    for ri, r in enumerate(rho):
        for x, y in zip(*img.nonzero()):
            ar = np.arange(x-r, x+r+1)
            for a in ar[ (ar >= 0) & (ar < shape[0]) ]:
                h = np.sqrt(r**2 - (a-x)**2)
                count(vote, ri, a, y - np.floor(h))
                count(vote, ri, a, y + np.floor(h))
                if h != np.floor(h):
                    count(vote, ri, a, y - np.floor(h+1))
                    count(vote, ri, a, y + np.floor(h+1))
    return vote
def fuzzy(votes, a=2.0, sigma=1.5, R=2):
    def kernel(k,sigma,R=2):
        kern = np.zeros(2*R+1)
        for d in range(0,R+1):
            value = k * np.exp(-d**2/sigma**2)
            kern[R+d] = value
            kern[R-d] = value
        return kern
    kern = kernel(a, sigma, R)
    fv = np.zeros(votes.shape)
    xx, yy = np.mgrid[0:fv.shape[1], 0:fv.shape[2]]
    for i, j in np.vstack([xx.ravel(), yy.ravel()]).T:
        fv[:,i,j] = np.convolve(votes[:,i,j], kern, mode='same')
    return fv

def fuzzyHT(f0, rmin, rmax):
    def kernel(k,sigma,R=2):
        kern = np.zeros(2*R+1)
        for d in range(0,R+1):
            value = k * np.exp(-d**2/sigma**2)
            kern[R+d] = value
            kern[R-d] = value
        return kern
    rr = range(rmin, rmax)
    v = HT(f0, rr)
    kern = kernel(1.0, 6)
    fv = np.zeros(v.shape)
    xx, yy = np.mgrid[0:f0.shape[0], 0:f0.shape[1]]
    for i, j in np.vstack([xx.ravel(), yy.ravel()]).T:
        fv[:,i,j] = np.convolve(v[:,i,j], kern, mode='same')
    return v, fv

def ring(f0, y0, x0, r):
    """ Return ring radius and thickness

        calculate the radial distance to (x0, y0) for every pixel in region,
        then fit the histogram of the distances to gauss function 
    """
    from scipy.optimize import curve_fit
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    dr = int(0.5*r)
    f1 = f0[y0-r-dr:y0+r+dr+1, x0-r-dr:x0+r+dr+1] 

    # calculate distance    
    dst = []
    for (x,y), p in np.ndenumerate(f1):
        if p > 0:
            dst.append(np.sqrt((x-r-dr)**2 + (y-r-dr)**2))

    # fit histogram of dst[] to gauss function
    hist, bin_edges = np.histogram(dst, 2*r)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    coeff, var_matrix = curve_fit(gauss, bin_centers, hist, p0=[10,r,3])

    return coeff[0], coeff[1], (bin_centers[1]-bin_centers[0])*coeff[2], var_matrix, dst


def mask_transform(f, y0, x0, r, dr):
    import skimage.draw as draw
    _f0 =  np.zeros(f.shape)
    _f1 =  np.zeros(f.shape)
    for _ir in range(r-dr, r+dr+1):
        yy, xx = draw.circle_perimeter(y0, x0, _ir, 'andres')
        _f0[yy, xx] = 1
    _f1[np.where(np.logical_and(f, _f0))] = 1
    _v, _fv = fuzzyHT(_f1, range(10, 50))
    return _f1, _fv


def rad_hist(f0, y, x, r):
    dr = int(2*r)
    f1 = f0[y-r-dr:y+r+dr+1, x-r-dr:x+r+dr+1] 
    dst = []
    for (x,y), p in np.ndenumerate(f1):
        if p > 0:
            dst.append(np.sqrt((x-r-dr)**2 + (y-r-dr)**2))
    xx, yy = np.histogram(dst, np.arange(0, r+dr, 0.5))
    return 0.5 * (yy[:-1] + yy[1:]), xx

def fit_gauss(xx, yy, *p0):
    from scipy.optimize import curve_fit
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    coeff, var_matrix = curve_fit(gauss, xx, yy, p0=p0)
    return coeff

def find_ring(fv, alpha = 0.9):
    #from skimage.feature import peak_local_max
    #rings = []
    #for _i, _f in enumerate(fv):
        #_g = np.copy(_f)
        #_g[_g<alpha*_g.max()] = 0
        #_r = peak_local_max(_g)
        #rings.extend(_r)
    #return np.array(rings)
    r, y, x = np.where(fv.max()==fv)
    r = r[0]; y = y[0]; x = x[0]
    #xx, yy = rad_hist(f0, y, x, r)
    return r, y, x

def rm_ring(fv, f0, y0, x0, r0, rmin, rmax):
   # hough transfrom of the masked image
    #_, _fv = fuzzyHT(fmask, rmin, rmax)
    #fv_sub = fv - _fv
    #print pk[0]
    #print flag
    return fmask


def mask_ring(fv, f0, y0, x0, r0, rmin):
    import skimage.draw as draw
    _fvr = np.sum(np.sum(fv[:,y0-1:y0+2,x0-1:x0+2], axis=1), axis=1)
    pk = np.where(np.r_[False, _fvr[1:] > _fvr[:-1]] & np.r_[_fvr[:-1] > _fvr[1:], False])
    xx = range(len(_fvr))
    flag = np.zeros(len(_fvr), dtype='uint8')
    coeffs = []
    for p in pk[0]:
        if p > 2*r0:
            break
        yy = np.copy(_fvr)
        yy[:p-3] = 0; yy[p+4:] = 0
        cc = fit_gauss(xx, yy, _fvr[p], p, 1)
        flag[np.floor(cc[1]-cc[2]):np.ceil(cc[1]+cc[2])] = 1
        coeffs.append(cc)
    # create image mask
    mask = np.zeros(f0.shape, dtype='uint8')
    for _r, _f in enumerate(flag):
        if _f == 1:
             _y, _x = draw.circle_perimeter(y0, x0, _r+rmin, 'andres')
             mask[_y, _x] = 1
    fmask = np.logical_and(f0, mask)
    return fmask, coeffs
  
