import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import atEmbed, atMatrix


def getTransitionFromMem(mem, tau, N=None):
    '''Return the mu-forward the mu-backward transition matrices for a given lag tau from a membership vector.'''
    # Get the correlation matrix
    nt = mem.shape[0]
    if N is None:
        N = np.max(mem)
    C = np.zeros((N, N))
    for t in np.arange(nt-tau):
        C[mem[t], mem[t+tau]] += 1.
    C /= C.sum()
    
    # Get the initial and final distributions for the non-autonomous case
    # (both should coincide in the autonomous case)
    initDist = C.sum(1)
    finalDist = C.sum(0)
    P = atMatrix.toLeftStochastic(C)
    Q = atMatrix.toLeftStochastic(C.T)

    return (P, Q, initDist, finalDist)


def getAdaptedRectGrid(ts, nx0, nSTD):
    '''Get a rectangular grid of nx0 boxes and spanning nSTD standard deviations of ts in each direction'''
    dim = ts.shape[0]
    nx = [nx0] * dim
    x = []
    xlims = []
    obsMean = ts.mean(1)
    obsSTD = ts.std(1)
    for d in np.arange(dim):
        x.append(np.linspace(obsMean[d] - obsSTD[d]*nSTD,
                             obsMean[d] + obsSTD[d]*nSTD, nx0))
        xlim = np.empty((nx0+1,))
        xlim[1:-1] = x[d][:-1] + (x[d][1] - x[d][0]) / 2
        xlim[0] = -1.e127
        xlim[-1] = 1.e127
        xlims.append(xlim)
    return (x, xlims)


def getBoxMemLevelsWeight(boxWeights, nLevels):
    '''Assign each grid-box to nLevels of equal weights by decreasing weight'''
    nBox = boxWeights.shape[0]
    targetWeights = np.ones((nLevels+1,)) * 1. / (nLevels+1)
    weights = np.zeros((nLevels+1,))
    mem = np.zeros((nBox,), dtype=int)
    isNotUsed = np.ones((nBox,), dtype=bool)
    nUsed = 0
    for k in np.arange(nLevels+1):
        while (weights[k] < targetWeights[k]) & (nUsed < nBox):
            iNotUsedMax = np.argmax(boxWeights[isNotUsed])
            iMax = np.arange(nBox)[isNotUsed][iNotUsedMax]
            weights[k] += boxWeights[iMax]
            mem[iMax] = k
            isNotUsed[iMax] = 0
            nUsed += 1
            
    return (mem, weights)


def getBoxMemSectors(positions, positionCenter, nSectors):
    '''Assign each grid-box to nSectors sectors'''
    nBox = positions.shape[1]
    sectorsLim = np.linspace(0., 2*np.pi, nSectors+1, endpoint=True)
    positionsComplex = positions[0] + 1j*positions[1]
    positionCenterComplex = positionCenter[0] + 1j*positionCenter[1]
    angles = np.angle(positionsComplex - positionCenterComplex) % (2*np.pi)
    mem = np.empty((nBox,), dtype=int)
    for k in np.arange(nSectors):
        mem[(angles >= sectorsLim[k]) & (angles < sectorsLim[k+1])] = k

    return mem


def getGridMem(ts, xlim):
    '''Get membership of each realization of ts in box of limits xlim'''
    (dim, nt) = ts.shape
    nx = []
    for d in np.arange(dim):
        nx.append(xlim[d].shape[0] - 1)
    N = np.prod(nx)
    mem = np.ones((nt,), dtype=int)*(-1)
    # Iterate on all grid-boxes
    for box in np.arange(N):
        ## Find all the realizations in the box
        ### Initialize an array of realization for which each element
        ### will stay true if the corresponding realization is in the box
        realization = np.ones((nt,), dtype=bool) 
        ### Iterate in each direction
        subbp = box
        for ind in np.arange(dim):
            subbn = int(subbp/nx[ind])
            ids = subbp - subbn*nx[ind] # Index of box in each direction ind
            # For the realizations to be in the box, it should be in the
            # right interval in each direction (hence the loop)
            realization &= (ts[ind] >= xlim[ind][ids]) \
                           & (ts[ind] < xlim[ind][ids+1])
            subbp = subbn
        # Set the mapping to the grid box if any realization belongs to it
        if np.any(realization):
            mem[realization] = box
            
    return mem


def getBoxMemPolar(boxPositions, boxWeights, nLevels, nSectors, centerType):
    '''Assign boxes to levels, based on their weight, and sectors'''
    # Get levels based on the probability to be inside a contour.
    (memLevels, weights) = getBoxMemLevelsWeight(boxWeights, nLevels)

    # Get center
    if centerType == 'mode':
        idMode = np.argmax(boxWeights)
        centerPosition = boxPositions[:, idMode]
    elif centerType == 'mean':
        centerPosition = np.sum(boxPositions \
                                * np.tile(boxWeights,
                                          (boxPositions.shape[0], 1)))

    # Get sectors membership
    memSectors = getBoxMemSectors(boxPositions, centerPosition, nSectors)

    # Get complete membership
    memPolar = np.ones((boxPositions.shape[1],), dtype=int) * (-1)
    memCount = 0
    for lev in np.arange(nLevels+1):
        for sect in np.arange(nSectors):
            memPolar[(memLevels == lev) & (memSectors == sect)] = memCount
            memCount += 1

    return memPolar


def getPolarMem(ts, nLevels, nSectors, nx0=200, nSTD=6, rotation=None, 
                centerType='mode', bw_method='scott'):
    '''Assign levels and sectors to realization of time series ts'''
    N = (nLevels+1) * nSectors
    if rotation == 'eof':
        # EOF rotation
        (w, v, tsRot) = atEmbed.EOFRotation(ts)
    else:
        tsRot = ts

    # Get adapted rectangular grid
    (x, xlims) = getAdaptedRectGrid(tsRot, nx0, nSTD)
    (X, Y) = np.meshgrid(x[0], x[1])
    boxPositions = np.vstack([X.flatten(), Y.flatten()])

    # Get membership of realizations to boxes
    memReaGrid = getGridMem(tsRot, xlims)

    # Get 2D PDF
    pdfKernel = stats.gaussian_kde(tsRot, bw_method=bw_method)
    boxWeights = pdfKernel(boxPositions)
    Z = np.reshape(boxWeights, X.shape) \
        * (x[0][1] - x[0][0])*(x[1][1] - x[1][0])
    boxWeights = Z.flatten()
    #boxWeights = np.zeros((nx0**2,))
    #for k in np.arange(nx0**2):
    #    boxWeights[k] = np.sum(memReaGrid == k)
    #boxWeights /= np.sum(memReaGrid >= 0)

    # Get membership of boxes to levels and sectors
    boxMemPolar = getBoxMemPolar(boxPositions, boxWeights, nLevels, nSectors,
                                 centerType)
    polarLabels = np.unique(boxMemPolar)
    nPolar = polarLabels.shape[0]
    if nPolar < N:
        boxMemPolarOld = boxMemPolar.copy()
        boxMemPolar = np.empty(boxMemPolarOld.shape, dtype=int)
        for k in np.arange(nPolar):
            boxMemPolar[boxMemPolarOld == polarLabels[k]] = k
    polarLabels = np.unique(boxMemPolar)

    # Get membership of realizations to sectors
    reaMemPolar = boxMemPolar[memReaGrid]

    # Unrotate positions
    if rotation == 'eof':
        boxPositionsUnRot = np.dot(v.T, boxPositions)
    else:
        boxPositionsUnRot = boxPositions
    (XUnRot, YUnRot) = (boxPositionsUnRot[0].reshape(nx0, nx0),
                        boxPositionsUnRot[1].reshape(nx0, nx0))
    xPolar = []
    xpol = np.empty((nPolar,))
    ypol = np.empty((nPolar,))
    for k in np.arange(nPolar):
        xpol[k] = np.mean(boxPositionsUnRot[0][boxMemPolar == polarLabels[k]])
        ypol[k] = np.mean(boxPositionsUnRot[1][boxMemPolar == polarLabels[k]])
    xPolar.append(xpol)
    xPolar.append(ypol)

    return (reaMemPolar, boxMemPolar, XUnRot, YUnRot, xPolar)


    
