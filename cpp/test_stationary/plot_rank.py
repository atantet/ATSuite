import os
import numpy as np
from scipy import sparse
import atgraph, atgraph_sparse, atplot, atmath, atcom
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# Define pdf domain
nx = 53
xbound = np.linspace(-0.015, 0.01, nx+1)
#nx = 42
#xbound = np.linspace(-0.01, 0.01, ny+1)
#nx = 35
#xbound = np.linspace(-0.015, 0.01, nx+1)

ny = 42
ybound = np.linspace(-0.01, 0.01, ny+1)
#ny = 28
#ybound = np.linspace(-0.01, 0.01, ny+1)
N = nx * ny
x = (xbound[1:] + xbound[:-1]) / 2
dx = (xbound[1:] - xbound[:-1]) / 2
y = (ybound[1:] + ybound[:-1]) / 2
dy = (ybound[1:] - ybound[:-1]) / 2
(X, Y) = np.meshgrid(x, y)
(DX, DY) = np.meshgrid(dx, dy)
(XL, YL) = (X.flatten(), Y.flatten())
(DXL, DYL) = (DX.flatten(), DY.flatten())
(XLmDXL, YLmDYL, XLpDXL, YLpDYL) = (XL - DXL, YL - DYL, XL + DXL, YL + DYL)

src_cmap = cm.hot_r
connected = np.loadtxt("/Users/atantet/PhD/dev/bve_t21/transfer/pc01/graph/connected_nt73000_N2226_dt09.txt", dtype=bool)
id_connected = np.nonzero(connected)[0]    
    
# Plot stationary distribution (PageRank)
rank_phase = np.loadtxt('rank.txt')
fig = plt.figure()
rank_all = np.ma.masked_all((N,), dtype=rank_phase.dtype)
rank_all[connected] = rank_phase
plt.pcolormesh(X, Y, rank_all.reshape(ny, nx),
               cmap=src_cmap)
cbar = plt.colorbar()
