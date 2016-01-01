def community_filter_sub(adj, mem_filt, min_den, icom):
    # Remove sparsely interconnected nodes from each community
    iout = np.zeros((len(mem_filt),), dtype=bool)
    coms = np.unique(mem_filt)
    for ii in icom:
        # Nodes of the community ii
        inodes = mem_filt == coms[ii]
        # Nodes removed from community ii
        Nci = np.sum(inodes)
        # If only one or no nodes, leave the loop
        Nci0 = Nci
        if Nci <= 1:
            continue
        # Find the node with the smallest degree
        #        mod_inodes = adj_n2c[ind_inodes, ii] - np.sum(adj_n2c[:, ii]) * np.sum(adj_n2c[ind_inodes], axis = 1)
        # Select adjacency matrix of community ii
        iout_ii = iout[inodes]
        adj_ii = adj[np.ix_(inodes, inodes)]
        # Calculate the iner-degree of nodes of the community
        sum_adj_ii = np.sum(adj_ii, axis=1)
        # Find the minimum degree
        amin_mod = np.argmin(sum_adj_ii)
        # Remove nodes while the minimum degree is lower than the average 
        # degree vertices of a community with density min_den would have
        not_iout = ~iout_ii
        ind_not_iout = np.where(not_iout)[0]
        while sum_adj_ii[ind_not_iout[amin_mod]] < min_den * (Nci - 1):
            # If the degree of the node is smaller than min_den, remove it and continue
            iout_ii[ind_not_iout[amin_mod]] = True
            Nci -= 1
            if Nci <= 1:
                break
            rm = adj_ii[ind_not_iout[amin_mod]].astype(bool)
            sum_adj_ii[rm] -= 1
            #            mod_inodes = adj_n2c[ind_inodes, ii] - np.sum(adj_n2c[:, ii]) * np.sum(adj_n2c[ind_inodes], axis=1)
            not_iout = ~iout_ii
            # index in ii
            ind_not_iout = np.where(not_iout)[0]
            # Look for the index of the minimum degree in the nodes
            # which haven't been removed
            amin_mod = np.argmin(sum_adj_ii[not_iout])
        iout[inodes] = iout_ii
    return iout


def community_filter_weak(adj, membership, min_den, min_size, nproc=1):
# remove nodes connected to less than min_den nodes of each community and communities
# smaller than min_size times the total number of nodes in the network
# Parallel version
    # Allocation and dimensions
    mem_filt = membership.copy()
    # The minimul community number must be 1, 0 is reserved for orphans
    mem_filt -= np.min(mem_filt) - 1
    coms = np.unique(mem_filt)
    ncom = coms.shape[0]
    N = len(mem_filt)
    rm = 0.

    # Test is more tasks than processes
    if ncom < nproc:
        nproc = ncom
    # Open the pool
    res = [None] * nproc
    if nproc > 1:
        pool = mp.Pool(processes=nproc)
        
        # Remove sparsely connected nodes
        for k in range(nproc):
            icom = range(k, ncom, nproc)
            res[k] = pool.apply_async(community_filter_sub, 
                                      args=(adj, mem_filt, min_den, icom))
        pool.close()
        pool.join()
        for k in range(nproc):
            iout = res[k].get()
            rm += np.sum(iout)
            mem_filt[iout] = 0
    else:
        iout = community_filter_sub(adj, mem_filt,
                                    min_den, min_size, range(ncom))
        rm += np.sum(iout)
        mem_filt[iout] = 0

    # Filter the remaining communities
    mem_filt = community_filter_size(mem_filt, min_size)

    return mem_filt


