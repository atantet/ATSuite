#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy.sparse.linalg
from scipy import sparse
from atmath import find_first
import atMatrix
import igraph

def get_adjacency(similarity, threshold, edge_density=None):
    adj = (similarity >= threshold) - np.eye(similarity.shape[0], dtype=bool)
    return adj


def area_weighted_connectivity(graph, lat):
    weight = np.cos(lat * np.pi / 180)
    norm = weight.sum()
    awc = np.array(graph.degree() * weight / norm)
    return awc


def get_edge_density_function(sim, nbins):
    # May differ from igraph.density() if sim has NaNs
    # Select values for j > i
    triu_ind = np.triu_indices_from(sim, k=1)
    sim1d = sim[triu_ind[0], triu_ind[1]]
    sim1dnn = np.abs(sim1d[~np.isnan(sim1d)])
    
    rng = (0., np.max(sim1dnn))
    # Estimate normalized histogram
    (hist, threshold) = np.histogram(sim1dnn, range=rng, bins=nbins)
    hist = hist.astype("float64")
    hist /= len(sim1dnn)
    link_density_function = np.empty(nbins)
        
        #  Calculate the link density function
    for i in xrange(nbins):
        link_density_function[i] = hist[i:].sum()
        
    return (link_density_function, threshold)


def get_cross_degree(adjacency):
    N = adjacency.shape[0]
    degree1 = np.empty((N/2,), dtype=float)
    degree2 = np.empty((N/2,), dtype=float)
    for ii in range(N/2):
        degree1[ii] = np.sum(adjacency[ii, N/2:N-1])
        degree2[ii] = np.sum(adjacency[N/2+ii, 0:N/2-1])
    return (degree1, degree2)


def get_threshold_from_density(sim, rho):
    #  Flatten and sort correlation measure matrix
    N = sim.shape[0]
    flat_sim = sim.copy()
    flat_sim = flat_sim.flatten()
    flat_sim.sort()
        
    threshold = flat_sim[int((1 - rho) * (N**2 - N))]
    return threshold


def community_filter(membership, tau, rank=None):
    mem_filt = membership.copy()
    N = membership.shape[0]
    if rank is None:
        rank = np.ones((N,)) * 1. / N
        
    ucom = np.unique(mem_filt)
    coms = ucom[np.where(ucom > 0)]
    
    # Remove communities smaller than tau
    for ii in coms:
        inodes = mem_filt == ii
        if np.sum(rank[inodes]) < tau:
            mem_filt[inodes] = 0
            
    # Find the remainging communities
    ucom = np.unique(mem_filt)
    coms = ucom[np.where(ucom > 0)]
    ncom = len(coms)
    # Reduce indices
    for ii in range(ncom):
        mem_filt[mem_filt == coms[ii]] = ii + 1

    return mem_filt


def community_rank(membership, rank=None):
    N = membership.shape[0]
    # If rank is None rank is uniform so as to return community connectivity
    if rank is None:
        rank = np.ones((N,)) * 1. / N
    coms = np.unique(membership)
    ncom = coms.shape[0]

    rank_com = np.zeros((ncom,))
    for k in range(ncom):
        inodes = membership == coms[k]
        rank_com[k] = rank[np.nonzero(inodes)].sum()

    return rank_com


def get_com_ts(ts, membership):
    # Get mean time series of communities
    # Attention: Nodes of membership 0 are not considered part of a community
    nt = ts.shape[0]
    nnodes = ts.shape[1]
    # Mask invalid
    ts_ma = np.ma.masked_where(np.abs(ts) > 1.e6, ts)
    ncom = np.max(membership)
    ts_com = np.empty((nt, ncom))
    for jj in range(ncom):
        jj_nodes = np.where(membership == jj + 1)[0]
        ts_com[:, jj] = np.ma.mean(ts_ma[:, jj_nodes], axis = 1)
        del jj_nodes
    return ts_com


def get_adj_com_corr(ts, membership):
    ts_com = get_com_ts(ts, membership)
    adj_com = np.corrcoef(ts_com.T)
    return adj_com


def read_tree(tree_file):
    # Reads the output of the infomap algorithm as a tree
    f = open(tree_file, "r")
    # Read header line
    columns = f.readline().split()
    # Read code-length
    code_len = float(columns[2])
    # Read number of nodes
    N = int(columns[8])

    # Allocate rank vector
    rank = np.empty((N,), dtype=float)
    
    # Read first line of membership to detect number of levels
    columns = f.readline().split()
    # Extract content
    levels = columns[0].split(':')
    # The integer after the last comma is the rank within the module,
    # not the last module!
    nlevels = len(levels)
    membership = np.ones((N, nlevels), dtype=int)

    node_id = int(columns[2].strip('"')) - 1
    rank[node_id] = float(columns[1])
    membership[node_id, :] = levels[:nlevels]

    for k in range(1, N):
        # Read line
        columns = f.readline().split()
        # Extract content
        levels = columns[0].split(':')
        nlevels = len(levels)
        # Reallocate if more levels
        if nlevels > membership.shape[1]:
            membership_old = membership.copy()
            membership = np.ones((N, nlevels), dtype=int)
            membership[:, :membership_old.shape[1]] = membership_old
            del membership_old
        node_id = int(columns[2].strip('"')) - 1
        rank[node_id] = float(columns[1])
        membership[node_id, :nlevels] = levels[:nlevels]

    # Make sublevel communities unique
    for lev in range(1, membership.shape[1]):
        # Communities of the above (coarser) level
        parent_coms = np.sort(np.unique(membership[:, lev - 1]))
        for k in range(1, len(parent_coms)):
            # Number of childrien in leading parent community
            incr = np.max(membership[membership[:, lev - 1] == parent_coms[k - 1], lev])
            membership[membership[:, lev - 1] == parent_coms[k], lev] += incr

    f.close()
    return (membership.astype(int), rank, code_len)


def node2node_flow_from_rank(adj, rank):
# Get flow matrix of the network from the PageRank
# and degree of its nodes
    N = adj.shape[0]
    T = atMatrix.toRightStochastic(adj)
    flow = np.zeros((N, N))
    for ii in range(N):
        for jj in range(N):
            flow[ii, jj] = T[ii, jj] * rank[ii]
    return flow


def node2com_flow_from_flow(flow, member, direction='to'):
    coms = np.unique(member)
    ncom = coms.shape[0]
    N = member.shape[0]
    n2c_flow = np.empty((N, ncom), dtype=float)
    for k in range(ncom):
        inodes = member == coms[k]
        # First neighbours
        if direction == 'to':
            n2c_flow[:, k] = np.sum(flow[:, inodes], 1)
        elif direction == 'from':
            n2c_flow[:, k] = np.sum(flow[inodes, :], 0)
        else:
            sys.exit('Direction must be "to" or "from".')

    return n2c_flow


def node2com_flow_from_trans(T, member, dist=None, direction='to'):
    T = np.matrix(atMatrix.toRightStochastic(T))
    N = T.shape[0]
    if dist is None:
        dist = pagerank(T)
    dist = np.matrix(dist).reshape(N, 1)
    coms = np.unique(member)
    ncom = coms.shape[0]
    if direction == 'to':
        n2c_flow = np.matrix(np.empty((N, ncom)))
    elif direction == 'from':
        n2c_flow = np.matrix(np.empty((ncom, N)))
    for k in range(ncom):
        inodes = member == coms[k]
        if direction == 'to':
            n2c_flow[:, k] = np.multiply(T[:, inodes].sum(1), dist)
        elif direction == 'from':
            n2c_flow[k, :] = np.multiply(T[inodes, :], np.tile(dist[inodes], (1, N))).sum(0)
        else:
            sys.exit('Direction must be "to" or "from".')
    return n2c_flow


def node2com_trans_from_trans(T, member, dist=None, direction='to'):
    T = np.matrix(atMatrix.toRightStochastic(T))
    N = T.shape[0]
    if (dist is None) and (direction == 'from'):
        dist = pagerank(T)
    dist = np.matrix(dist).reshape(N, 1)
    coms = np.unique(member)
    ncom = coms.shape[0]
    if direction == 'to':
        n2c_trans = np.matrix(np.empty((N, ncom)))
    elif direction == 'from':
        n2c_trans = np.matrix(np.empty((ncom, N)))
    for k in range(ncom):
        inodes = member == coms[k]
        if direction == 'to':
            n2c_trans[:, k] = T[:, inodes].sum(1)
        elif direction == 'from':
            n2c_trans[k, :] = np.multiply(T[inodes, :], np.tile(dist[inodes], (1, N))).sum(0) / dist[inodes].sum()
        else:
            sys.exit('Direction must be "to" or "from".')
    return n2c_trans


def com2com_flow_from_flow(flow, member):
# Get the flow matrix of the module partition from
# the PageRank, Degree and membership of its nodes
    coms = np.unique(member)
    ncom = len(coms)
    com_flow = np.zeros((ncom, ncom))

    # Calculate inter-community flow
    for ii in range(ncom):
        ii_com = member == coms[ii]
        for jj in range(ncom):
            jj_com = member == coms[jj]            
            com_flow[ii, jj] = np.sum(flow[np.ix_(ii_com, jj_com)])
    return com_flow

    
def com2com_flow_from_rank(adj, rank, member):
# Get the flow matrix of the module partition from
# the PageRank, Degree and membership of its nodes
    coms = np.unique(member)
    ncom = len(coms)
    com_flow = np.zeros((ncom, ncom))

    # Calculate node flow
    flow = get_flow_from_rank(adj, rank)

    # Calculate inter-community flow
    for ii in range(ncom):
        ii_com = member == coms[ii]
        for jj in range(ncom):
            jj_com = member == coms[jj]            
            com_flow[ii, jj] = np.sum(flow[np.ix_(ii_com, jj_com)])
    return com_flow

    
def com2com_flow_from_map(map_file):
# Get the flow matrix of the module parition from
# the map file output of the Infomap algorithm
    # Open map file and read all lines
    f = open(map_file, 'r')
    
    # Find Modules section
    k = 0
    line = f.readline()
    while line[:8] != '*Modules':
        line = f.readline()
        k += 1
    ncom = int(line.split()[1])
    flow = np.zeros((ncom, ncom))
    # Read Modules section
    for ii in range(ncom):
        line = f.readline()
        flow[ii, ii] = float(line.split()[2])
    
    # Find Links section
    k = 0
    line = f.readline()
    while line[:6] != '*Links':
        line = f.readline()
        k += 1
    # Number of links
    E = int(line.split()[1])

    # Read Links section
    for ii in range(E):
        line = f.readline()
        aline = line.split()
        ci = int(aline[0]) - 1
        cj = int(aline[1]) - 1
        flow[ci, cj] = float(aline[2])
    f.close()
    return flow


def get_gradient_similarity(field, dist, dtype=None, precision=1.):
    # Radius of the earth to calculate the arc length
    N = field.shape[0]
    if dtype is None:
        dtype = field.dtype
    sim = np.zeros((N, N), dtype=dtype)
    for ii in range(N):
        for jj in range(ii+1, N):
            sim[ii, jj] = 1. * np.abs(field[jj] - field[ii]) / dist[ii, jj]
            sim[jj, ii] = sim[ii, jj]
    sim = np.round(sim * precision).astype(dtype)
    return sim


def get_flow_adjacency(stream, lon, lat, loc_param, glob_param=None):
    # Give the adjacency matrix of flow network from field (directed, unweighted)
    N = stream.shape[0]
    adj = np.zeros((N, N), dytpe=logical)
    loc_p = np.sqrt(loc_param / np.pi)
    u, v = atmath.get_vel_from_stream(stream, lon, lat)
    if glob_param is None:
        use_glob = False
    for ii in range(N):
        for jj in range(N):
            dist = np.sqrt(mod(lon[ii] - lon[jj], 180.)**2 + mod(lat[ii] - lat[jj], 180.)**2)
            # Local link definition
            is_loc = dist <= (loc_p * (u**2 + v**2)**(1. / 4))
            # Global link definition
            # TODO with contour
            is_glob = False
            # Define link
            adj[ii, jj] = is_loc | (use_glob & is_glob)
    return adj

def weighted_degree(weighted_adj, inward=False, norm=False):
    # Return the degree or the out-degree for a directional network
    # unless inward is True, in which case the in-degree is returned
    axis = int(not inward)
    degree = np.sum(weighted_adj, axis).astype(float)
    if norm:
        degree = degree / np.sum(weighted_adj)
    return degree

def degree_proj_xy(adj, angle):
    # angle must be in radians
    Xdeg = sum(np.abs(adj * np.cos(angle)), 0)
    Ydeg = sum(np.abs(adj * np.sin(angle)), 0)
    return (Xdeg, Ydeg)

def write_edgelist(adj, dst_path):
    adj_type = adj.dtype
    fmt = '%d %d '
    if (adj.dtype == int) or (adj.dtype == bool):
        fmt = fmt + '%d'
    else:
        fmt = fmt + '%e'
    fmt = fmt + '\n'
        
    N = adj.shape[0]
    f = open(dst_path, 'w')

    # Write list of edges
    ind = np.nonzero(adj)
    E = ind[0].shape[0]
    print 'Writing %d edges to %s...' % (E, dst_path)
    for k in range(E):
        f.write(fmt % (ind[0][k] + 1, ind[1][k] + 1,
                       adj[ind[0][k], ind[1][k]]))
    f.close()

def writePajekFromAdj(adj, dst_path):
    # Write pajek for an integer adjacency matrix
    adj_type = adj.dtype
    fmt = '%d %d '
    if (adj.dtype == int) or (adj.dtype == bool):
        fmt = fmt + '%d'
    else:
        fmt = fmt + '%e'
    fmt = fmt + '\n'
        
    N = adj.shape[0]
    f = open(dst_path, 'w')

    # Write list of vertices
    print 'Writing %d vertices to %s...' % (N, dst_path)
    f.write('*Vertices %d\n' % N)
    for k in range(N):
        f.write('%d "%d"\n' % (k+1, k+1))

    # Write list of edges
    ind = np.nonzero(adj)
    E = ind[0].shape[0]
    print 'Writing %d edges to %s...' % (E, dst_path)
    f.write('*Edges %d\n' % E)
    for k in range(E):
        f.write(fmt % (ind[0][k] + 1, ind[1][k] + 1, adj[ind[0][k], ind[1][k]]))
    f.close()
    
def read_pajek(path, dtype=float, start_zero=False):
    substr = 1
    if start_zero:
        substr = 0

    # Open Pajek file
    f = open(path, 'r')

    # Get number of vertices
    line = f.readline()
    N = int(line.split()[1])
    data = np.empty((N, N), dtype=dtype)
    
    line = f.readline()
    while line[:6] != '*Edges':
        line = f.readline()

    # Read edge list
    E = int(line.split()[1])
    for k in range(E):
        line = f.readline()
        arr_line = line.split()
        id_i = int(arr_line[0]) - substr
        id_j = int(arr_line[1]) - substr
        data[id_i, id_j] = dtype(arr_line[2])
    f.close()
    return data


def ts2AdjPajek(var, tau, giant=False):
    """ Build a Pajek graph from time series using correlations. The weight is unweighted and undirected, but edges are repeated."""
    N = var.shape[1]
    edges = []
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            corr = np.corrcoef(var[:, i], var[:, j])[0, 1]
            if np.abs(corr) > tau:
                edges.append((i, j))
                edges.append((j, i))
    graph = igraph.Graph(N, directed=True)

    return graph
    
def ts2AdjPajekCut(var, tau, nCut=1, giant=False, verbose=False):
    """ Build a Pajek graph from time series using correlations. The weight is unweighted and undirected, but edges are repeated."""
    (T, N) = var.shape

    if verbose:
        print 'Normalize...'
    mu = var.mean(0)
    sigma = var.std(0)
    for k in np.arange(T):
        var[k] = (var[k] - mu) / sigma
    
    srcNodes = np.array([], dtype=int)
    dstNodes = np.array([], dtype=int)
    for i in np.arange(nCut):
        for j in np.arange(nCut):
            #            corrCut = np.corrcoef(var[:, N/nCut*i:N/nCut*(i+1)],
            #                                  var[:, N/nCut*j:N/nCut*(j+1)], rowvar=False)
            if verbose:
                print 'Get cross correllation for block (%d, %d)...' % (i, j)
            corrCut = np.dot(var[:, N/nCut*i:N/nCut*(i+1)].T,
                             var[:, N/nCut*j:N/nCut*(j+1)]) / T
            if verbose:
                print 'Get adjacency block...'
            adjCut = np.abs(corrCut) > tau
            # Remove diagonal
            if i == j:
                for k in np.arange(adjCut.shape[0]):
                    adjCut[k, k] = False
            if verbose:
                print 'Update edge list...'
            nzCut = np.nonzero(adjCut)
            srcNodes = np.concatenate((srcNodes, nzCut[0] + N/nCut*i))
            dstNodes = np.concatenate((dstNodes, nzCut[1] + N/nCut*j))
    if verbose:
        print 'Build graph...'
    edges = map(tuple, np.concatenate((srcNodes, dstNodes)).reshape(2, len(srcNodes)).T)
    graph = igraph.Graph(N, edges, directed=True)

    return graph
    
def local_entropy(adj, inward=False):
    # Normalize adjacency
    N = adj.shape[0]
    if inward:
        adj_norm = adj.astype(float).T
    else:
        adj_norm = adj.astype(float)
    for ii in range(N):
        norm = np.sum(adj_norm[ii])
        if norm != 0:
            adj_norm[ii] = adj_norm[ii] / norm
    # Cacluate entropy
    plogp = np.log(adj_norm)
    plogp[np.isinf(plogp)] = 0.
    plogp *= adj_norm
    s = -np.sum(plogp, 1)
    return s


# def pagerank(adj, d=0.85, tol=0):
#     N = adj.shape[0]

#     # Get transition matrix
#     T = atMatrix.toRightStochastic(adj)

#     # Uniform transition matrix
#     U = np.ones((N, N)) / N
    
#     # Get damped transition matrix
#     Td = d * T + (1. - d) * U

#     # Get First Eigen-Vector
#     x = np.ones((N,)) * 1. / N
#     xi = np.dot(x, Td)
#     k = 1
#     eps = np.sum((xi - x)**2)
#     while eps > tol**2:
#         x = xi.copy()
#         xi = np.dot(x, Td)
#         k += 1
#         eps = np.sum((xi - x)**2)
    
#     return xi


def pagerank(T, tol=0, maxiter=0):
    N = T.shape[0]

    # Get transition matrix
    T = atMatrix.toRightStochastic(T)

    # Get First Eigen-Vector
    x = np.matrix(np.ones((N,)) * 1. / N)
    xi = x * T
    k = 1
    eps = np.abs(xi - x).sum()
    iteration = 0
    while eps > tol:
        x = xi.copy()
        xi = x * T
        k += 1
        eps = np.abs(xi - x).sum()
        iteration += 1
        if (maxiter > 0) and (iteration >= maxiter):
            break
    
    return xi

def arnoldi(T, k=2, d=0.85, tol=0):
    N = T.shape[0]
    if sparse.issparse(T):
        Tdense = T.todense()
    else:
        Tdense = T

    # Uniform transition matrix
    U = np.ones((N, N)) / N
    
    # Get damped transition matrix
    Td = d * Tdense + (1. - d) * U

    # Solve eigen-value problem for the left eigen-vectors
    (w, v) = scipy.sparse.linalg.eigs(Td.T, k=k, tol=tol)
    
    return (w, v.T)
    
def reducedMarkovMember(T, member, p=None):
    m = T.shape[0]
    if p is None:
        p = pagerank(T, tol=1.e-10)
    p = np.squeeze(np.array(p))
    invs = np.unique(member)
    n = invs.shape[0]
    
    M = np.zeros((m, n), dtype=float)
    L = np.zeros((n, m), dtype=float)
    mu = np.empty((n,), dtype=float)
    
    for alpha in np.arange(n):
        imem = member == invs[alpha]
        M[:, alpha] = imem
        mu[alpha] = p[imem].sum()
        if mu[alpha] > 0:
            L[alpha, imem] = M[imem, alpha] * p[imem] / mu[alpha]

    R = np.matrix(L) * T * np.matrix(M)

    return (R, mu)
    
def reducedMarkov(T, M, p=None):
    T = np.matrix(T)
    M = np.matrix(M)
    (m, n) = M.shape
    if p is None:
        p = pagerank(T, tol=1.e-10)

    mu = p * M

    diagP = np.matrix(np.diag(np.squeeze(np.array(p))))
    iDiagMu = np.matrix(np.diag(1. / np.squeeze(np.array(mu))))

    L = iDiagMu * M.T * diagP

    R = L * T * M

    return (R, mu)


def reverseMarkov(T, p=None):
    T = np.matrix(T)
    if p is None:
        p = pagerank(T, tol=1.e-10)
    p = np.squeeze(np.array(p))
    D = np.matrix(np.diag(p))
    DI = np.matrix(np.diag(1. / p))
    RT = DI * T.T * D
    symT = (T + RT) / 2
    return symT


def symMarkov(T, p=None):
    T = np.matrix(T)
    if p is None:
        p = pagerank(T, tol=1.e-10)
    p = np.squeeze(np.array(p))
    D = np.matrix(np.diag(np.sqrt(p)))
    DI = np.matrix(np.diag(np.sqrt(1. / p)))
    symT = D * T.T * DI
    return symT

def coo2pajek(coo, dst_path, start_zero=False):
    substr = 1
    if start_zero:
        substr = 0
    
    # Write pajek for an integer adjacency matrix
    coo_type = coo.dtype
    fmt = '%d %d '
    if (coo.dtype == int) or (coo.dtype == bool):
        fmt = fmt + '%d'
    else:
        fmt = fmt + '%e'
    fmt = fmt + '\n'
        
    N = coo.shape[0]
    f = open(dst_path, 'w')

    # Write list of vertices
    f.write('*Vertices %d\n' % N)
    for k in range(N):
        f.write('%d "%d"\n' % (k+substr, k+substr))

    # Write list of edges
    E = coo.nnz
    f.write('*Edges %d\n' % E)
    for k in range(E):
        f.write(fmt % (coo.row[k] + substr, coo.col[k] + substr, coo.data[k]))
    f.close()
    

def edgeList2pajek(dst_path, N, row, col, weight=None, start_zero=False):
    substr = 1
    if start_zero:
        substr = 0
    
    # Write pajek for an integer adjacency matrix
    fmt = '%d %d '
    if weight is not None:
        if (weight.dtype == int) or (weight.dtype == bool):
            fmt = fmt + '%d'
        else:
            fmt = fmt + '%e'
    fmt = fmt + '\n'
        
    f = open(dst_path, 'w')

    # Write list of vertices
    f.write('*Vertices %d\n' % N)
    for k in range(N):
        f.write('%d "%d"\n' % (k+substr, k+substr))

    # Write list of edges
    E = row.shape[0]
    f.write('*Edges %d\n' % E)
    for k in range(E):
        if weight is not None:
            f.write(fmt % (row[k] + substr, col[k] + substr, weight[k]))
        else:
            f.write(fmt % (row[k] + substr, col[k] + substr))
    f.close()
    

def coo2pajekSym(coo, dst_path, start_zero=False):
    substr = 1
    if start_zero:
        substr = 0
    
    # Write pajek for an integer adjacency matrix
    coo_type = coo.dtype
    fmt = '%d %d '
    if (coo.dtype == int) or (coo.dtype == bool):
        fmt = fmt + '%d'
    else:
        fmt = fmt + '%e'
    fmt = fmt + '\n'
        
    N = coo.shape[0]
    f = open(dst_path, 'w')

    # Write list of vertices
    f.write('*Vertices %d\n' % N)
    for k in range(N):
        f.write('%d "%d"\n' % (k+substr, k+substr))

    # Write list of edges
    E = 0
    for k in range(coo.nnz):
        if coo.row[k] > coo.col[k]:
            E += 1
    f.write('*Edges %d\n' % E)
    for k in range(coo.nnz):
        if coo.row[k] > coo.col[k]:
            f.write(fmt % (coo.row[k] + substr, coo.col[k] + substr, coo.data[k]))
    f.close()
    

def pajek2coo(path, dtype=float, start_zero=False):
    substr = 1
    if start_zero:
        substr = 0

    # Open Pajek file
    f = open(path, 'r')

    # Get number of vertices
    line = f.readline()
    N = int(line.split()[1])

    # Go to Edges section
    line = f.readline()
    while line[:6] != '*Edges':
        line = f.readline()

    # Read edge list
    E = int(line.split()[1])
    i = np.empty((E,), dtype=dtype)
    j = np.empty((E,), dtype=dtype)
    data = np.empty((E,), dtype=dtype)
    for k in range(E):
        line = f.readline()
        arr_line = line.split()
        i[k] = int(arr_line[0]) - substr
        j[k] = int(arr_line[1]) - substr
        data[k] = dtype(arr_line[2])
    f.close()
    coo = sparse.coo_matrix((data, (i, j)), shape=(N, N))
    return coo


def strength(spmat, inward=False):
    N = spmat.shape[0]
    if inward:
        T = spmat.transpose()
        if type(T) != sparse.csr_matrix:
            T = T.tocsr()
    else:
        T = spmat
    strength = T.sum(1)
    return strength


def degree(spmat, inward=False):
    N = spmat.shape[0]
    degree = np.zeros((N,), dtype=int)
    if inward:
        T = spmat.transpose()
        if type(T) != sparse.csr_matrix:
            T = T.tocsr()
    else:
        T = spmat
    
    for k in range(N):
        degree[k] = T.indptr[k+1] - T.indptr[k]

    return degree
    

def local_entropy(spmat, inward=False):
    # Normalize adjacency
    N = spmat.shape[0]
    s = np.zeros((N,))
    if inward:
        T = spmat.transpose()
        if type(T) != sparse.csr_matrix:
            T = T.tocsr()
    else:
        T = spmat

    for k in range(N):
        data =  T[k].data
        norm = data.sum()
        if norm > 0:
            Tn = data * 1. / norm
            # Calculate entropy
            s[k] = -(Tn * np.log(Tn)).sum()

    return s


def closeness(spmat):
    N = spmat.shape[0]
    dist_mat = sparse.csgraph.dijkstra(spmat)
    dist_mat[np.isinf(dist_mat)] = N
    closeness = 1. / np.mean(dist_mat, axis=1)
    return closeness


def com2comTrans(T, member):
# Get the flow matrix of the module partition from
# the PageRank, Degree and membership of its nodes
    coms = np.unique(member)
    ncom = coms.shape[0]
    com2com = np.zeros((ncom, ncom))

    if sparse.issparse(T):
        for ii in range(ncom):
            iiCom = member == coms[ii]
            Tii = T[iiCom]
            if T.format == 'csr':
                rowIndices = Tii.indices
            # elif T.format == 'lil':
            #     rowIndices = Tii.rows[0]
            else:
                sys.exit('Type %s of sparse matrix not implemented.' % T.format)
            for k in range(Tii.nnz):
                jj = rowIndices[k]
                com2com[ii, np.nonzero(coms == member[jj])] += Tii.data[k]
    com2com = atMatrix.toRightStochasticCSR(sparse.csr_matrix(com2com))

    return com2com

   
def node2com_flow_from_flow(flow, member, direction='to'):
    # Returns the node to community flow or the node too community connectivity if an adjacency matrix is given
    if direction == 'to':
        F = flow.transpose()
        if type(F) != sparse.csr_matrix:
            F = F.tocsr()
    else:
        F = flow
    coms = np.unique(member)
    ncom = coms.shape[0]
    N = member.shape[0]
    if (F.dtype == int) or (F.dtype == bool):
        dtype = int
    else:
        dtype = float
    n2c_flow = np.zeros((N, ncom), dtype=dtype)
    for ii in range(ncom):
        # Attention: element access with bools does not work with sparse
        #            matrices, use np.nonzero()
        ii_com = np.nonzero(member == coms[ii])[0]
        flow_ii = F[ii_com]
        for k in range(flow_ii.nnz):
            n2c_flow[flow_ii.indices[k], ii] += flow_ii.data[k]
    return n2c_flow


def node2node_flow_from_rank(T_csr, rank):
# Get flow matrix of the network from the PageRank
# and degree of its nodes
    N = T_csr.shape[0]
    rank_csr = sparse.diags(rank, 0, format='csr')
    flow_csr = rank_csr * T_csr

    return flow_csr


def pagerank(adj_csr, tol=0):
    N = adj_csr.shape[0]

    # Get transition matrix
    T = atMatrix.toRightStochasticCSR(adj_csr)

    # Get First Eigen-Vector
    x = np.matrix(np.ones((N,)) * 1. / N)
    xi = x * T
    k = 1
    eps = np.abs(xi - x).sum()
    while eps > tol:
        x = xi.copy()
        xi = x * T
        k += 1
        eps = np.abs(xi - x).sum()
    
    return np.array(xi)

