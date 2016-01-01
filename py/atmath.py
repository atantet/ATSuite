#  Import numpy
import numpy as np
from scipy import stats, interpolate, linalg, misc, sparse
import sys
import multiprocessing as mp

# Calculate cross correlation
def cross_corr(X, Y, lags=[0], masked=False):
    """X: numpy array of size (ntimes, nnodes)
    Y:idem
    lags: list of lags"""
    s = np.shape(X)
    # Create cross correlation array
    cc = np.empty((len(lags), s[0], s[0]), dtype = 'float32')
    if np.shape(Y) != s:
        sys.exit("X and Y must have the same size.")

    # Loop on the lags
    for k in range(len(lags)):
        lag = lags[k]
	# Create troncated (lagged) matrices
        Xl = X[:, 0:s[1]-lag]
        Yl = Y[:, lag:s[1]]
	# Use numpy corrcoef to compute correlations and select upper right block of the correlation matrix
        if masked:
            cc[k, :, :] = np.ma.corrcoef(Xl, Yl)[0:s[0], s[0]:2*s[0]]
        else:
            cc[k, :, :] = np.corrcoef(Xl, Yl)[0:s[0], s[0]:2*s[0]]
    return cc


# get_anomaly
def get_anomaly(observable, time_cycle):
    anomaly = np.empty(observable.shape)
    
    for i in range(time_cycle):
        anomaly[i::time_cycle] = observable[i::time_cycle] - observable[i::time_cycle].mean(axis = 0)
    return anomaly


# reg2lin
def reg2lin(lat, lon):
    """Convert regular grid lat/lon vectors to curvilinear grid (list)
    """
    nlat = len(lat)
    nlon = len(lon)
    latlin = np.empty((nlat * nlon, 1))
    lonlin = np.empty((nlat * nlon, 1))
    for ii in range(nlat):
        for jj in range(nlon):
            latlin[nlon * ii + jj] = lat[ii]
            lonlin[nlon * ii + jj] = lon[jj]
    return (latlin, lonlin)


def select_months(observable, selected_months, time_cycle=12):
    # Only works for monthly averages
    # Only takes full years
    # Must begin in January
    
    s = observable.shape
    # Number of years
    range_years = int(s[0] / time_cycle)

    # Create array of indices by month of the year for each year
    phase_indices = np.zeros((time_cycle, range_years), dtype=int)

    # Sort indices by month of the year for each year
    for i in range(time_cycle):
        phase_indices[i, :] = np.arange(i, range_years * time_cycle, time_cycle)
    
    #  Select time indices corresponding to chosen phase indices
    selected_indices = phase_indices[selected_months, :]

    #  Flatten and sort selected time indices
    selected_indices = selected_indices.flatten()
    selected_indices.sort()

    return selected_indices


def iecdf(x, p, nbins=10):
    """f = iecdf(x, p, nbins=10) returns the reciprocal of the empirical cumulative distriution function at ordinate p
    """
    # if (p > 1 or p < 0):
    #     print "Error : Percentile p must be between 0 and 1."
    #     exit
    cum = stats.cumfreq(x, nbins)
    a = cum[0] / len(x)
    lowlim = cum[1]
    bsize = cum[2]
    uplim = lowlim + bsize * nbins
    bins = np.linspace(lowlim + bsize / 2, uplim - bsize / 2, nbins)
    freqs = interpolate.interp1d(a, bins)
    f = freqs(p)
    return f

def mi(x, y, bins):
# Rewritten by Qingyi Feng

    (N, n_samples) = x.shape
    
    #  Initialize mutual information array
    mi = np.zeros((N, N))
    px = np.zeros((N, bins))
    py = np.zeros((N, bins))
    
    
    #  Get common range for all histograms
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    
    #  Calculate the histograms for each time series
    for i in xrange(N):
        px[i,:] = (np.histogram(x[i, :], bins=bins, range=(x_min,x_max))[0]).astype("float64")
    
    px /= n_samples
    
    #  Make sure that bins with zero estimated probability are not counted 
    #  in the entropy measures.
    px[px == 0] = 1
    
    #  Compute the information entropies of each time series
    Hx = - (px * np.log(px)).sum(axis = 1)
    
    
    #  Calculate the histograms for each time series
    for i in xrange(N):
        py[i,:] = (np.histogram(y[i, :], bins=bins, range=(y_min,y_max))[0]).astype("float64")
    
    py /= n_samples
    
    #  Make sure that bins with zero estimated probability are not counted 
    #  in the entropy measures.
    py[py == 0] = 1
    
    #  Compute the information entropies of each time series
    Hy = - (py * np.log(py)).sum(axis = 1)
    
    for i in xrange(N):
        for j in xrange(N):
            #  Calculate the joint probability distribution
            pxy = (np.histogram2d(x[i,:], y[j,:], bins=bins, range=((x_min, x_max), (y_min, y_max)))[0]).astype("float64")
            
            #  Normalize joint distribution
            pxy /= n_samples
            
            #  Compute the joint information entropy
            pxy[pxy == 0] = 1
            Hxy = - (pxy * np.log(pxy)).sum()
            
            #  ... and store the result
            mi.itemset((i,j), Hx.item(i) + Hy.item(j) - Hxy)
    
    return mi

def mi(x, bins):
# Rewritten by Qingyi Feng

    (N, n_samples) = x.shape
    
    #  Initialize mutual information array
    mi = np.zeros((N, N))
    px = np.zeros((N, bins))
    
    
    #  Get common range for all histograms
    x_min = x.min()
    x_max = x.max()
    
    #  Calculate the histograms for each time series
    for i in xrange(N):
        px[i,:] = (np.histogram(x[i, :], bins=bins, range=(x_min,x_max))[0]).astype("float64")
    
    px /= n_samples
    
    #  Make sure that bins with zero estimated probability are not counted 
    #  in the entropy measures.
    px[px == 0] = 1
    
    #  Compute the information entropies of each time series
    Hx = - (px * np.log(px)).sum(axis = 1)
    
    
    for i in xrange(N):
        for j in xrange(N):
            #  Calculate the joint probability distribution
            pxx = (np.histogram2d(x[i,:], x[j,:], bins=bins, range=((x_min, x_max), (x_min, x_max)))[0]).astype("float64")
            
            #  Normalize joint distribution
            pxx /= n_samples
            
            #  Compute the joint information entropy
            pxx[pxx == 0] = 1
            Hxx = - (pxx * np.log(pxx)).sum()
            
            #  ... and store the result
            mi.itemset((i,j), Hx.item(i) + Hx.item(j) - Hxx)
    
    return mi


def varimax(x, normalize=False, tol=1.e-5, it_max=1000):
# Translation of R varimax algorithm
# Usage: new_loads, rotmax = varimax(loadings, normalize, tolerance, it_max )
# See Sherin, 1966 for the matrix formulation of the problem
    # Get number of random variables and loadings
    (nvar, nload) = x.shape
    if nload < 2:
        return
    L = x.copy()
    
    if normalize:
        factor = np.tile(np.sqrt(np.diag(np.dot(L, L.T))), (nload, 1)).T
        L = L / factor
        
    # First approximate the rotated loadings by the orignial loadings
    # i.e. set rotation matrix to identity
    R = np.eye(nload)
    d = 0.

    for ii in range(it_max):
        # Rotate the loadings
        B = np.dot(L, R)
        
        # Matrix form of the varimax problem
        PmS = np.dot(L.T, (B**3 - np.dot(B, np.diagflat(np.dot(np.ones((1, nvar)), B**2))) / nvar))
  
        # Decompose PmS using SVD (in numpy: PmS = U * S * V)
        (U, S, V) = linalg.svd(PmS)
  
        # Calculate rotation matrix from SVD eigen-vectors (see matrix formulation of the pb)
        # R = (PmS * PmS')^(-1/2) * PmS
        #   = (U * (S * S') * U')^(-1/2) * U S V
        #   = U * (S * S')^(-1/2) * U' * U S V
        #   = U * V
        R = np.dot(U, V)

        dpast = d
        d = np.sum(S)
        
        # End if exceeded tolerance.
        if d < dpast * (1. + tol):
            break
    if ii < it_max - 1:
        print "Varimax: convergence to %8.2e at iteration %d." % (tol, ii)
    else:
        print "Varimax has not converged to %8.2e because maximum iteration %d \
        was reached." % (tol, it_max)
    # Final matrix.
    B = np.dot(L, R)

    # Renormalize.
    if normalize:
        B = B * factor

    return (B, R)


def perms(v):
    v = np.unique(v)
    N = v.shape[0]
    if N == 1:
        return v
    p = np.empty((misc.factorial(N, exact=1), N), dtype=v.dtype)
    npprev = misc.factorial(N - 1, exact=1)
    for k in range(N):
        p[k*npprev:(k+1)*npprev, 0] = v[k]
        p[k*npprev:(k+1)*npprev, 1:] = perms(v[v != v[k]])
    return p

# def combs(v, npicks):
#     v = np.unique(v)
#     N = v.shape[0]
#     if npick == 1:
#         return v
#     c = np.empty((misc.comb(N, npicks, exact=1), npicks), dtype=v.dtype)
#     npprev = misc.comb(

def unique_rows(X):
    uni_X = np.array([np.array(x) for x in set(tuple(x) for x in X)],
                     dtype=X.dtype)
    return uni_X


def lag_autocorr(ts, d=1):
    # Deal with missing values
    if np.ma.isMaskedArray(ts):
        ad = lag_autocorr_ma(ts, d)
    else:
        N = ts.shape[0]
        X = ts[:N-d].astype(float)
        # Innovation
        Xi = ts[d:].astype(float)
        # Remove sample means
        X = X - np.mean(X)
        Xi = Xi - np.mean(Xi)
        # Valid data
        ad = np.dot(X, Xi) / np.sqrt(np.dot(X, X) * np.dot(Xi, Xi))
    return ad


def lag_autocorr_ma(ts, d=1):
    # Deal with missing values
    N = ts.shape[0]
    X = ts[:N-d].astype(float)
    # Innovation
    Xi = ts[d:].astype(float)
    # Valid data
    and_valid = (~X.mask) & (~Xi.mask)
    X_v = X.data[and_valid]
    Xi_v = Xi.data[and_valid]
    # Remove sample mean
    X_v = X_v - np.mean(X_v)
    Xi_v = Xi_v - np.mean(Xi_v)
    # Calculate autocorrelation as Pearson
    ad = np.dot(X_v, Xi_v) / np.sqrt(np.dot(X_v, X_v) * np.dot(Xi_v, Xi_v))
    # TODO: Calculate as AR(1)
    return ad

def lag_autocorr_vect(X, d=1, nproc=1):
    Ntasks = X.shape[1]
    a1 = np.empty((Ntasks,))
    # Define the number of iterations by worker
    if Ntasks < nproc:
        # Fewer tasks than processes
        nproc = Ntasks

    # Open the pool
    res = [None] * nproc
    pool = mp.Pool(processes=nproc)
    
    # Launch processes
    for k in range(nproc):
        rng = range(k, Ntasks, nproc)
        res[k] = pool.apply_async(lag_autocorr_vect_mono, args=(X, d, rng,))
    # Close and join the pool
    pool.close()
    pool.join()

    # Compile results
    for k in range(nproc):
        rng = range(k, Ntasks, nproc)
        iout = res[k].get()
        a1[rng] = iout

    return a1

def lag_autocorr_vect_mono(X, d, rng):
    Ntasks = len(rng)
    a1 = np.empty((Ntasks,))
    for k in range(Ntasks):
        a1[k] = lag_autocorr(X[:, rng[k]], d)
    return a1

def zonal_angle(lonL, latL):
    N = lonL.shape[0]
    ang = np.zeros((N, N))
    for ii in range(0, N):
        for jj in range(ii+1, N):
            ang[ii, jj] = np.arctan2((latL[jj] - latL[ii]),
                                     (lonL[jj] - lonL[ii]))
            ang[jj, ii] = ang[ii, jj]
    return ang

def geo_dist(lon, lat, arc=False):
    N = lon.shape[0]
    # Earth radius in meters
    Rearth = 6371009
    dist = np.zeros((N, N), dtype=lon.dtype)
    if arc:
        # Convert coordinates in radians
        lon = lon * np.pi / 180.
        lat = lat * np.pi / 180.
        for ii in range(N):
            sinlatii = np.sin(lat[ii])
            coslatii = np.cos(lat[ii])
            for jj in range(ii+1, N):
                # Arc length between the two points
                dist[ii, jj] = np.arccos(sinlatii * np.sin(lat[jj])
                                         + coslatii * np.cos(lat[jj])
                                         * np.cos(lon[jj] - lon[ii])) * Rearth
                dist[jj, ii] = dist[ii, jj]
    else:
        for ii in range(N):
            for jj in range(ii+1, N):
                dist[ii, jj] = np.sqrt((lat[jj] - lat[ii])**2
                                       + np.mod(lon[jj] - lon[ii], 180.)**2)
                dist[jj, ii] = dist[ii, jj]
    # Return distance in meters
    return dist


def find_first(a, value=None):
    if len(a.shape) > 1:
        a = a.flatten()
    if value is None:
        if a.dtype is bool:
            value = False
        elif a.dtype is int:
            value = 0
        elif a.dtype is float:
            value = 0.
    for k in range(a.shape[0]):
        if value == a[k]:
            return k
            break
    return None


def plog2p(p):
    if p == 0.:
        return 0.
    else:
        return p * np.log2(p)

def entropy(p):
    p = np.squeeze(np.array(p))
    entropy = 0.
    for i in np.arange(p.shape[0]):
        entropy -= plog2p(p[i])
    return entropy

def entropyRate(p, T):
    p = np.squeeze(np.array(p))
    entropyRate = 0.
    for i in np.arange(T.shape[0]):
        for j in np.arange(T.shape[1]):
            entropyRate -= p[i] * plog2p(T[i, j])
    return entropyRate

def mutualInformation(p, T):
    MI = entropy(p) - entropyRate(p, T)
    return MI
    
def logDetEntropy(variance):
    entropy = 1. / 2 * np.log2(2*np.pi*np.e * variance)
    return entropy

def ccf(ts1, ts2, lagMax=None, sampFreq=1.):
    """ Cross-correlation function"""
    ts1 = (ts1 - ts1.mean()) / ts1.std()
    ts2 = (ts2 - ts2.mean()) / ts2.std()
    nt = ts1.shape[0]
    if lagMax is None:
        lagMax = nt - 1
    lagMaxSample = int(lagMax * sampFreq)
    ccf = np.empty((lagMaxSample*2+1,))
    for k in np.arange(lagMaxSample):
        ccf[k] = (ts1[:-(lagMaxSample-k)] * ts2[lagMaxSample-k:]).mean()
    ccf[lagMax] = (ts1 * ts2).mean()
    for k in np.arange(lagMaxSample):
        ccf[2*lagMaxSample-k] = (ts2[:-(lagMaxSample-k)] \
                                 * ts1[lagMaxSample-k:]).mean()
    return ccf


def ccovf(ts1, ts2, lagMax=None):
    """ Cross-covariance function"""
    ts1 = ts1 - ts1.mean()
    ts2 = ts2 - ts2.mean()
    nt = ts1.shape[0]
    if lagMax is None:
        lagMax = nt - 1
    ccovf = np.empty((lagMax*2+1,))
    for k in np.arange(lagMax):
        ccovf[k] = (ts1[:-(lagMax-k)] * ts2[lagMax-k:]).mean()
    ccovf[lagMax] = (ts1 * ts2).mean()
    for k in np.arange(lagMax):
        ccovf[2*lagMax-k] = (ts2[:-(lagMax-k)] * ts1[lagMax-k:]).mean()
    return ccovf

def getPerio(ts, freq=None, sampFreq=1., tapeWindow=None):
    ''' Get the periodogram of ts using a taping window of length tape window'''
    nt = ts.shape[0]

    # If no tapeWindow given then do not tape
    if tapeWindow is None:
        tapeWindow = nt
    nTapes = int(nt / tapeWindow)
    window = np.hamming(tapeWindow)

    # Get frequencies if not given
    if freq is None:
        freq = getFreqPow2(tapeWindow, sampFreq=sampFreq)
    nfft = freq.shape[0]

    # Get periodogram averages over nTapes windows
    perio = np.zeros((nfft,))
    perioSTD = np.zeros((nfft,))
    for tape in np.arange(nTapes):
        tsTape = ts[tape*tapeWindow:(tape+1)*tapeWindow] 
        tsTape -= tsTape.mean(0)
        tsWindowed = tsTape * window
        # Fourier transform and shift zero frequency to center
        fts = np.fft.fft(tsWindowed, nfft, 0)
        fts = np.fft.fftshift(fts)
        # Get periodogram
        perio += np.abs(fts)**2 / np.sum(np.abs(fts)**2)
        perioSTD += (np.abs(fts)**2 / np.sum(np.abs(fts)**2))**2
    perio /= nTapes
    perioSTD = np.sqrt(perioSTD / nTapes)

    return (freq, perio, perioSTD)

def getFreqPow2(nt, sampFreq=1., center=True):
    ''' Get frequency vector with maximum span given by the closest power of 2 (fft) of the lenght of the times series nt'''
    # Get nearest larger power of 2
    if np.log2(nt) != int(np.log2(nt)):
        nfft = 2**(int(np.log2(nt)) + 1)
    else:
        nfft = nt

    # Get frequencies
    freq = np.fft.fftfreq(nfft, d=1./sampFreq)

    # Shift zero frequency to center
    if center:
        freq = np.fft.fftshift(freq)

    return freq
