import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, patches, rcParams
from matplotlib.collections import PatchCollection


# Default parameters
levels = 20
fs_default = 'x-large'
fs_latex = 'xx-large'
fs_xlabel = fs_default
fs_ylabel = fs_default
fs_xticklabels = fs_default
fs_yticklabels = fs_default
fs_legend_title = fs_default
fs_legend_labels = fs_default
fs_cbar_label = fs_default
#            figFormat = 'eps'
figFormat = 'png'
dpi = 300
msize = 16
bbox_inches = 'tight'

def get_cmap_com(ncom, src_cmap=cm.gist_stern, ncol_rm=5, cmap_name='cmap_com', has_unassigned=True):
    first_cols = np.array([[1, 0, 0, 1], [0, 1, 0, 1],
                           [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                           [0, 0, 1, 1]])
    if has_unassigned:
        # Add white color for orphans
        first_cols = np.insert(first_cols, 0, [1, 1, 1, 1], axis=0)
    nfirst_cols = first_cols.shape[0]
    clist_orig = src_cmap(np.linspace(0, 1, ncom + ncol_rm))
    clist = np.empty((ncom, 4), dtype=int)
    if clist_orig.shape[0] - ncol_rm > 0:
        clist = clist_orig[:clist_orig.shape[0] - ncol_rm]
    k = 0
    while (k < ncom) and (k < nfirst_cols):
        clist[k] = first_cols[k]
        k += 1
    cmap_com = colors.LinearSegmentedColormap.from_list(cmap_name, clist, N=ncom)
    return cmap_com


def cbar_com(plot, collection, coms, rank_com=None, skip=1, has_unassigned=True, unassigned_name='Unassigned'):
    ncom = coms.shape[0]
    # ncom must count filtered communities
    srank = [''] * ncom
    if rank_com is not None:
        srank = [' - %4.1f%%' % (rank_com[k] * 100,) for k in range(ncom)]
    cbar = plot.colorbar(collection)
    collection.set_clim(vmin=coms[0], vmax=coms[-1])
    ticks = np.linspace(coms[0], coms[-1], ncom *2 + 1)
    cbar.set_ticks(ticks)
    cbar_labels = []
    # if has_unassigned:
    #     cbar_labels.append(orphans_name)
    #     kticks = ticks[1::skip]
    #     kadd = 0
    # else:
    #     kticks = ticks[0::skip]
    #     kadd = 1
    kticks = ticks[::skip]
    cbar_labels = [''] * len(ticks)
    rng_labels = np.arange(1, len(ticks) - 1, 2 * skip)
    for k in range(rng_labels.shape[0]):
        if k == 0 and has_unassigned:
            cbar_labels[rng_labels[k]] = unassigned_name + srank[k]
        else:
            cbar_labels[rng_labels[k]] = '%2u' % coms[k*skip] + srank[k]
    cbar.set_ticklabels(cbar_labels)
    cbar.ax.yaxis.set_ticks_position('none')


def pcolor_rectangle(x, y, dx, dy, data, vmin=None, vmax=None, cmap=None, ncolors=256, norm=None):
    if cmap is None:
        cmap = cm.get_cmap(rcParams['image.cmap'], ncolors)
    N = x.shape[0]
    patch = []
    for k in range(N):
        rect = patches.Rectangle((x[k] - dx[k] / 2., y[k] - dy[k] / 2.), dx[k], dy[k])
        patch.append(rect)
    pcollection = PatchCollection(patch, cmap=cmap, edgecolor='none', norm=norm)
    pcollection.set_array(data)
    pcollection.set_clim(vmin, vmax)
    return pcollection

def draw_default_map(map, lon_labels=[False, False, False, True],
                     lat_labels=[True, False, False, False],
                     lon_ticks=None, lat_ticks=None,
                     fill_cont=True, cont_color=None, lake_color='aqua'):
    if cont_color is None:
        cont_color = np.ones(3) * 0.8
    if lon_ticks is None:
        lon_ticks = np.arange(0.,360.,60.)
    if lat_ticks is None:
        lat_ticks = np.arange(-90.,120.,30.)
        
    map.drawcoastlines()
    if fill_cont:
        map.fillcontinents(color=cont_color, lake_color='aqua')
    map.drawparallels(lat_ticks, labels=lat_labels)
    map.drawmeridians(lon_ticks, labels=lon_labels)

def get_proj_coord(map, lon, lat, dlon, dlat):
    (crap, dy_low) = map(lon, lat - dlat / 2.)
    (crap, dy_up) = map(lon, lat + dlat / 2.)
    (dx_low, crap) = map(lon - dlon / 2., lat)
    (dx_up, crap) = map(lon + dlon / 2., lat)
    dx = dx_up - dx_low
    dy = dy_up - dy_low
    (x, y) = map(lon, lat)
    return (x, y, dx, dy)

def plotCCF(ccf, lags=None, ls='-', lc='b', lw=2, xlim=None, ylim=None,
            xlabel=None, ylabel=None, absUnit=''):
    '''Default plot for a correlation function.'''
    if lags is None:
        lags = np.arange(ccf.shape[0])
    if xlim is None:
        xlim = (0, lags[-1])
    if ylim is None:
        ylim = (-1.05, 1.05)
    if absUnit != '':
        absUnit = '(%s)' % absUnit
    if xlabel is None:
        xlabel = r'$t$ %s' % absUnit
    if ylabel is None:
        ylabel = r'$C_{x, x}(t)$'
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(lags, ccf, linestyle=ls, color=lc, linewidth=lw)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=fs_latex)
    ax.set_ylabel(ylabel, fontsize=fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)

    return (fig, ax)

def plotPerio(perio, freq=None, perioSTD=None, xlim=None, ylim=None,
              xlabel=None, ylabel=None, xscale='linear', yscale='linear',
              absUnit='', absType='ang', ls='-', lc='k', lw=2,
              fc=None, ec=None, alpha=0.2):
    '''Default plot for a periodogram.'''
    nfft = perio.shape[0]
    if freq is None:
        freq = np.arange(-nfft, nfft+1)
    if xlim is None:
        xlim = (0, freq[-1])
    if ylim is None:
        ylim = (perio.min(), perio.max())
    if absType == 'ang':
        absName = '\omega'
        absUnit = 'rad %s' % absUnit
    elif absType == 'freq':
        absName = 'f'
        absUnit = '%s' % absUnit
    if ylabel is None:
        ylabel = r'$\hat{S}_{x,x}(%s)$' % absName
    if absUnit != '':
        absUnit = '(%s)' % absUnit
    if fc is None:
        fc = lc
    if ec is None:
        ec = fc
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if perioSTD is not None:
        perioDown = perio - perioSTD / 2
        perioUp = perio + perioSTD / 2
        ax.fill_between(freq[nfft/2+1:], perioDown[nfft/2+1:],
                        perioUp[nfft/2+1:], facecolor=fc, alpha=alpha,
                        edgecolor=fc)
    ax.plot(freq[nfft/2+1:], perio[nfft/2+1:], linestyle=ls, color=lc,
            linewidth=lw)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$%s$ %s' % (absName, absUnit),
                  fontsize=fs_latex)
    ax.set_ylabel(ylabel, fontsize=fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)

    return (fig, ax)
