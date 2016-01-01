import numpy as np

def getEmbedding(ts, delays):
    '''Embed a multi-dimensional time-series'''
    delays0 = np.concatenate(([0], delays), 0)
    nDelays = len(delays0)
    (dim0, nt0) = ts.shape
    dim = dim0 * nDelays
    nt = nt0 - delays0[-1]

    tsEmbed = np.zeros((dim, nt))
    for k in np.arange(dim0):
        # The original one starts from the maximum lag
        tsEmbed[k] = ts[k, delays0[-1]:]
        for d in np.arange(1, nDelays):
            tsEmbed[dim0 * d + k] = ts[k, delays0[-(d+1)]:-delays0[d]]

    return tsEmbed

def EOFRotation(ts, corr=False):
    '''Get basis of EOFs and the associated principal components'''
    nt = ts.shape[1]

    # Remove means
    mean = ts.mean(1)
    ts -= np.tile(mean, (nt, 1)).T

    # Get similarity matrix
    if corr:
        ts /= np.tile(ts.std(1), (nt, 1)).T
    C = np.dot(ts, ts.T) / nt

    # Get EOFs
    (w, v) = np.linalg.eigh(C)
    isort = np.argsort(w)[::-1]
    w = w[isort]
    v = v[:, isort].T

    # Get principal components
    pc = np.dot(v, ts)

    return (w, v, pc)


