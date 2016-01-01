#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Import numpy
from sys import exit, maxint
import multiprocessing as mp
import numpy as np
from scipy import stats
import atmath

def mbb_bivar(statistic, parameter, param_args, X, Y=None, auto=False,
              L=None, Ns=2000, nproc=1):
    # Calculate statistic statistic of a bivariate random variable (X, Y)
    # from Moving Block Bootstrap of window length L and resamples set size Ns
    # on nproc processes. The statistic is calculated with arguments param_args.
    print 'Performing bootstrap on bivariate problem:'
    # If L is defined, be sure it is an integer
    vaX = None
    vaY = None
    if L is not None:
        L = int(L)
    # Define problem dimensions and allocate
    N1 = X.shape[1]
    is_sym = False
    if Y is None and auto:
        Ntasks = N1
        print 'Bivariate problem treated as univariate: Ntasks = %d' % Ntasks
        indices = np.empty((2, N1))
        indices[0] = np.arange(N1)
        indices[1] = np.arange(N1)
        Y = X
        if parameter == get_ci_from_set:
            param_res = np.zeros((N1, 2))
        else:
            param_res = np.zeros((N1,))
        if L is not None:
            lopt = L
        else:
            lopt = np.zeros((N1,), dtype=int)
            # Get autocorrelation used for lopt
            print 'Calculating auto-correlation at lag 1 for X...'
            vaX = atmath.lag_autocorr_vect(X, nproc=nproc)
            vaY = vaX

        estim = np.zeros((N1,))
    elif Y is not None and auto:
        exit('The bivariate statistic should not be estimated on only \
one variable if two are given.')
    else:
        if (Y is None and not auto) or np.array_equal(X, Y):
            # The problem is symetric
            N2 = N1
            Y = X
            Ntasks = N1 * (N1 - 1) / 2
            print 'Bivariate problem is symetric: Ntasks = %d(%d-1)/2 = %d' % (N1, N1, Ntasks)
            indices = np.triu_indices(N1, k=1)
            is_sym = True
            if L is not None:
                lopt = L
            else:
                lopt = np.zeros((N1, N2), dtype=int)
                # Get autocorrelation used for lopt
                print 'Calculating auto-correlation at lag 1 for X...'
                vaX = atmath.lag_autocorr_vect(X, nproc=nproc)
                vaY = vaX
        else:
            N2 = Y.shape[1]
            Ntasks = N1 * N2
            print 'X and Y differ: Ntasks = %dx%d = %d' % (N1, N2, Ntasks)
            indices = np.indices((N1, N2))
            if L is not None:
                lopt = L
            else:
                lopt = np.zeros((N1, N2), dtype=int)
                # Get autocorrelation used for lopt
                print 'Calculating auto-correlation at lag 1 for X and Y...'
                vaX = atmath.lag_autocorr_vect(X, nproc=int(nproc/2))
                vaY = atmath.lag_autocorr_vect(Y, nproc=int(nproc/2))
        if parameter == get_ci_from_set:
            # Two bounds for CI
            param_res = np.zeros((N1, N2, 2))
        else:
            param_res = np.zeros((N1, N2))

        estim = np.zeros((N1, N2))

    # Define the number of iterations by worker
    if Ntasks < nproc:
        # Fewer tasks than processes
        nproc = Ntasks

    # Open the pool
    res = [None] * nproc
    pool = mp.Pool(processes=nproc)
    
    # Launch processes
    print 'Lauching bootstraps...'
    for k in range(nproc):
        rng = range(k, Ntasks, nproc)
        res[k] = pool.apply_async(mbb_bivar_mono, 
                                  args=(statistic, parameter, param_args,
                                        X, Y, indices, Ns, rng, L, vaX, vaY))
    # Close and join the pool
    pool.close()
    pool.join()

    # Compile results
    for k in range(nproc):
        rng = range(k, Ntasks, nproc)
        iout = res[k].get()
        if auto:
            param_res[rng] = iout[0]
            estim[rng] = iout[1]
            if L is None:
                lopt[rng] = iout[2]
        else:
            param_res[indices[0][rng], indices[1][rng]] = iout[0]
            estim[indices[0][rng], indices[1][rng]] = iout[1]
            if L is None:
                lopt[indices[0][rng], indices[1][rng]] = iout[2]
            if is_sym:
                param_res[indices[1][rng], indices[0][rng]] = param_res[indices[0][rng], indices[1][rng]]
                estim[indices[1][rng], indices[0][rng]] = estim[indices[0][rng], indices[1][rng]]
                if L is None:
                    lopt[indices[1][rng], indices[0][rng]] = lopt[indices[0][rng], indices[1][rng]]

    return param_res, estim, lopt


def mbb_bivar_mono(statistic, parameter, param_args, X, Y, indices,
                   Ns, rng, L=None, vaX=None, vaY=None):
    # Solve MBB on one process
    Ntasks = len(rng)
    if parameter == get_ci_from_set:
        # Two values for CI
        res = np.zeros((Ntasks, 2))
    else:
        res = np.zeros((Ntasks,))
    if param_args[2] is not None:
        # Default Critical values allocation is infinite
        # and not zero as for P-values
        res = res * np.inf
    estim = np.zeros((Ntasks,))
    if L is not None:
        lopt = L
    else:
        lopt = np.zeros((Ntasks,), dtype=int)
    for k in range(Ntasks):
        # Allocate and define time series and autocorrelation
        set = np.empty((Ns,), dtype=float)
        rng_k = rng[k]
        x_all = X[:, indices[0][rng_k]]
        y_all = Y[:, indices[1][rng_k]]
        aX = vaX[indices[0][rng_k]]
        aY = vaX[indices[1][rng_k]]
        # Masked array?
        if np.ma.isMaskedArray(x_all):
            and_valid = (~x_all.mask) & (~y_all.mask)
            x = x_all.data[and_valid]
            y = y_all.data[and_valid]
        else:
            x = x_all
            y = y_all
        N = x.shape[0]
        # Calculate estimator (only used for P-Value and BCa CI)
        estim[k] = statistic(x, y)
        # What block length to use?
        if L is None:
            # Get optimal block length
            lopt[k] = lopt_equ_corr(aX, aY, N)
            L_use = lopt[k]
        else:
            L_use = L
        # Estimate parameter only if block length lower than length
        # of time series
        if np.abs(L_use) < N:
            # Set deck
            npick = N / L_use
            npick_rem = npick
            rem = np.mod(N, L_use)
            if rem > 0:
                npick_rem = npick + 1
            deck = range(0, N - L_use)
            # Define resample arguments
            resample_args = (x, y, L_use, npick, rem, deck, False)
            if parameter == get_ci_from_set:
                # Overtake resampling indices for CI
                resample_args[6] = True
                
            # Resample
            for s in range(Ns):
                # Get resamples
                (xp, yp) = block_resample_bivar(*resample_args)
                # Calculate bivariate function
                set[s] = statistic(xp, yp)
            # Get parameter result
            res[k] = parameter(set, *param_args, estim=estim[k], df=N)

    return (res, estim, lopt)


def block_resample(x, L, npick, rem, deck):
    # Create a surrogate ts from randomly picked overlapping blocks
    N = x.shape[0]
    Ldeck = len(deck)
    if rem > 0:
        npick_rem = npick + 1
    xp = np.empty((N,), dtype=type(x))
    irdn = np.random.randint(0, Ldeck, size=npick_rem)
    for k in range(npick):
        xp[k*L:(k+1)*L] = x[deck[irdn[k]]:deck[irdn[k]]+L]
    if rem > 0:
        xp[npick*L:] = x[deck[irdn[npick]]:deck[irdn[npick]]+rem]
        
    return xp

def block_resample_bivar(x, y, L, npick, rem, deck, overtake=False):
    # Create a surrogate ts from randomly picked overlapping blocks
    # If y is not given, create a set from x only
    N = x.shape[0]
    Ldeck = len(deck)
    if rem > 0:
        npick_rem = npick + 1
    xp = np.empty((N,), dtype=x.dtype)
    yp = np.empty((N,), dtype=y.dtype)
    irdn1 = np.random.randint(0, Ldeck, size=npick_rem)
    # Sould the x and y be resampled the same way?
    if overtake:
        irdn2 = irdn1
    else:
        irdn2 = np.random.randint(0, Ldeck, size=npick_rem)
    # Resample
    for k in range(npick):
        ideck1 = deck[irdn1[k]]
        ideck2 = deck[irdn2[k]]
        xp[k*L:(k+1)*L] = x[ideck1:ideck1+L]
        yp[k*L:(k+1)*L] = y[ideck2:ideck2+L]
    if rem > 0:
        ideck1 = deck[irdn1[npick]]
        ideck2 = deck[irdn2[npick]]
        xp[npick*L:] = x[ideck1:ideck1+rem]
        yp[npick*L:] = y[ideck2:ideck2+rem]
    return (xp, yp)


def get_sl_from_set(set, sl_type='percentile', use_fisher=False, alpha=None,
                    estim=None, df=0):
    # Get significance level at alpha from a random set
    # Fisher transformation converges faster (n) to normal distribution
    # The absolute value is taken and only the upper bound is given, which
    # could be problematic for skewed distributions
    # If significance level alpha is None then return P-value of estim
    if alpha is None and estim is None:
        exit('Error: at least significance level alpha or estimator estim required to return a critical value or a P-value.')
    Ns = set.shape[0]
    set = np.abs(set)
    if use_fisher:
        set = np.arctanh(set)

    # Normal SL 
    if sl_type == 'normal':
        if alpha is None:
            res = 1. - stats.norm.cdf(estim)
        else:
            res = stats.norm.ppf(1 - alpha)
    # Student's SL (good theoretical coverage but erratic in practice,
    # Efron and Tibshirani 1993)
    elif sl_type == 'student':
        if alpha is None:
            res = 1. - stats.t.cdf(estim, df)
        else:
            res = stats.t.ppf(1 - alpha, df)
    # Percentile SL (less erratic than t, but less good coverage, idem)
    elif sl_type == 'percentile':
        if alpha is None:
            res = 1. * np.sum(set > estim) / Ns
        else:
            res = np.sort(set)[np.ceil((1 - alpha) * Ns)]
    # Bias-corrected and accelerated SL (transformation respecting contrary
    # to t, second-order accurate contrary to percentile, idem)
    # TODO (but not the best)
    # elif sl_type == 'bca':
    #     # Bias correction (corrects for the median estimation bias)
    #     estim_mean = np.mean(set)
    #     # Used for significance level where bootstrap estimator
    #     # aren't estimates of the proper parameter (dirty)
    #     estim = estim_mean
    #     lower = np.sum(set < estim).astype(float) / Ns
    #     bc = stats.norm.ppf(lower)
    #     # Acceleration (takes into account scale effects arising when the
    #     # se of the estimator depends on the value of theta, the estimated.
    #     a = np.sum((estim_mean - set)**3) / \
    #                np.sum((estim_mean - set)**2)**(3./2) / 6
    #     norm_ub = stats.norm.ppf(1 - alpha)
    #     alpha2 = stats.norm.cdf(bc + (bc + norm_ub) / (1 - a * (bc + norm_ub)))
    #     level = np.sort(set)[np.min((np.ceil(alpha2 * Ns), Ns - 1))]
    else:
        exit('Error: sl_type given to get_sl_from_set does not correspond \
to any type of confidence interval.')
        
    if use_fisher and alpha is not None:
        res = np.tanh(res)
    return res


def get_ci_from_set(set, alpha, use_fisher=False,
                    ci_type='percentile', estim=0., df=0):
    # Get confidence interval at alpha from a random set
    Ns = set.shape[0]
    if use_fisher:
        set = np.arctanh(set)
        estim = np.arctanh(estim)
        
    # Normal CI
    if ci_type == 'normal':
        lb = stats.norm.ppf(alpha / 2)
        ub = stats.norm.ppf(1 - alpha / 2)
    # Student CI
    elif ci_type == 'student':
        lb = stats.t.ppf(alpha / 2, df)
        ub = stats.t.ppf(1 - alpha / 2, df)
    # Percentile CI
    elif ci_type == 'percentile':
        lb = np.sort(set)[np.floor(alpha / 2 * Ns)]
        ub = np.sort(set)[np.min((np.ceil((1 - alpha / 2) * Ns), Ns - 1))]
    # Bias-Corrected and accelerated
    elif ci_type == 'bca':
        # Bias correction (corrects for the median estimation bias)
        lower = np.sum(set < estim).astype(float) / Ns
        bc = stats.norm.ppf(lower)
        # Acceleration (takes into account scale effects arising when the
        # se of the estimator depends on the value of theta, the estimated.
        estim_mean = np.mean(set)
        a = np.sum((estim_mean - set)**3) / \
                   np.sum((estim_mean - set)**2)**(3./2) / 6
        norm_lb = stats.norm.ppf(alpha / 2)
        norm_ub = stats.norm.ppf(1 - alpha / 2)
        alpha1 = stats.norm.cdf(bc + (bc + norm_lb) / (1 - a * (bc + norm_lb)))
        alpha2 = stats.norm.cdf(bc + (bc + norm_ub) / (1 - a * (bc + norm_ub)))
        lb = np.sort(set)[np.floor(alpha1 / 2 * Ns)]
        ub = np.sort(set)[np.min((np.ceil(alpha2 * Ns), Ns - 1))]
    else:
        exit('Error: ci_type given to get_ci_from_set does not correspond \
to any type of confidence interval.')
    if use_fisher:
        lb = np.tanh(lb)
        ub = np.tanh(ub)
    # returns the width of the interval
    return np.array([lb, ub])

def get_bi_from_uni(param, method='max'):
    N = param.shape[0]
    param_bi = np.zeros((N, N), dtype=param.dtype)
    if method == 'max':
        for ii in range(N):
            for jj in range(ii + 1, N):
                param_bi[ii, jj] = np.max((param[ii], param[jj]))
                param_bi[jj, ii] = param_bi[ii, jj]
    else:
        exit('No method called ' + method + ' to augment univariate parameter\
 to bivariate.')
    return param_bi

        
def persist_time(ts, d=1, from_autocorr=False):
# Least squares estimator of the persistence time (Gaussian AR(1))
    if from_autocorr:
        ad = ts
    else:
        ad = atmath.lag_autocorr(ts, d)
    tau = - 1. * d / np.log(ad)
    return tau


def lopt_equ_a1(a1, N):
    # Estimation of the optimal block length from the lag-1 auto-correlation
    lopt = np.round((6.**(1./2) * a1 / (1. - a1**2))**(2./3) * N**(1./3)).astype(int)
    if lopt == 0:
        lopt = 1
    return lopt

def lopt_equ_corr(a1x, a1y, N):
    # Manage negative and infinite cases
    if (a1x >= 1.) or (a1y >= 1.):
        lopt = N - 1
    elif (a1x <= 0 and a1y > 0):
        lopt = lopt_equ_a1(a1y, N)
    elif (a1y <= 0 and a1x > 0):
        lopt = lopt_equ_a1(a1x, N)
    elif a1x <= 0 and a1y <= 0:
        lopt = 1
    else:
        a1 = np.sqrt(a1x * a1y)
        lopt = lopt_equ_a1(a1, N)
    
    return lopt

def lopt_equ_mudelsee(x, y, d=1.):
    tau1 = persist_time(x, d)
    tau2 = persist_time(y, d)
    lopt = 4 * np.max((x, y))

    return lopt


def at_corr(X, Y):
    # Used for correlation bootstrap
    return np.corrcoef(X, Y)[0, 1]


def MBBCCF(x, y, lagMax, sampFreq, Ns, width=None):
    nt = x.shape[0]
    nWindows = int(nt / width)

    # Get CCF
    ccf = atmath.ccf(x, y, lagMax=lagMax, sampFreq=sampFreq)
    
    # Get window width (Sherman et al 1998, Mudelsee 2010)
    if width is None:
        eTimeSample = np.nonzero(ccf > 1. / e)[0][-1] + 1
        a = np.exp(-1./eTimeSample)
        width = int((6**(1./2)*a/(1-a**2))**(2./3) * nt**(1./3))
