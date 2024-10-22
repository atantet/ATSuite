import numpy as np
from scipy import sparse
import atgraph_sparse, atmath, atgraph

# Code length calculation
def twolevelCodelengthFromTrans(T_csr, member, u=None, Hu=None, uMod=None, TMod=None):
    if u is None:
        # Compute stationnary distrigution
        (crap, u) = atgraph.arnoldi(T_csr, k=1)
        u = np.abs(u).T
        u_csr = sparse.csr_matrix(u / u.sum())
    else:
        u_csr = sparse.csr_matrix(u)

    if Hu is None:
        # Compute the entropy of the stationary distribution
        Hu = atmath.entropy(u_csr)

    # Compute the entropy of the stationnary distribution of the partition
    if uMod is None:
        uMod_csr = sparse.csr_matrix(atgraph.community_rank(member, u_csr))
    else:
        uMod_csr = sparse.csr_matrix(uMod).T
    HQ = atmath.entropy(uMod_csr)

    # Get the total inter-module entropy
    HMod = Hu - HQ

    # Get the parition transition matrix
    if TMod is None:
        TMod_csr = atgraph_sparse.com2comTrans(T_csr, member)
    else:
        TMod_csr = sparse.csr_matrix(TMod)

    # Get the entropy of the partition's dynamics
    logTMod_csr = TMod_csr.copy()
    logTMod_csr.data = logTMod_csr.data * np.log2(logTMod_csr.data)
    hQM = (-uMod_csr * logTMod_csr.sum(1))[0, 0]

    # Get descritiption code length
    L = HMod + hQM
    
    return (L, Hu, HQ, hQM)


# Greedy search
def greedyCodelength(T):
    if not sparse.issparse(T):
        print 'Converting matrix to LIL...'
        T_lil = sparse.lil_matrix(T)
    elif T.format != 'lil':
        print 'Converting matrix to LIL...'
        T_lil = sparse.lil_matrix(T)
    else:
        T_lil = T
    T_csr = T_lil.tocsr()

    # Initialization
    print 'Initializing...'
    N = T_lil.shape[0]

    # Initialize membership vector
    member = np.arange(N)

    # Get stationnary distribution and its entropy
    (crap, u) = atgraph.arnoldi(T_lil, k=1)
    u = np.abs(u).T
    u_lil = sparse.lil_matrix(u / u.sum())
    Hu = atmath.entropy(u_lil)
    print 'Initial entropy of stationnary distribution = %f' % Hu

    # Get initial codelength (the entropy of the Markov process)
    logT_csr = T_csr.copy()
    logT_csr.data = logT_csr.data * np.log2(logT_csr.data)
    hM = (-u_lil.T * logT_csr.sum(1))[0, 0]
    print 'Initial codelength or entropy rate is %f' % hM

    modIndex = np.unique(member)
    nMod = modIndex.shape[0]

    TCom_lil = T_lil.copy()
    uCom_lil = u_lil.copy()

    (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u_lil, Hu=Hu, uMod=u_lil, TMod=T_csr)
    
    print Hu
    print HQ
    print hQM
    print L

    # Loop on the number of iteration
    codelength = [hM] 
    nIter = 1
    while nMod > 1:
        # Search
        codeMin = Hu * 10
        argCodeMin = (0, 0)
        for ii in range(N):
            print 'Row %d' % ii
            nnzRow = TCom_lil[ii].nnz
            TComRow = TCom_lil[ii].rows[0]
            memberWork = member.copy()
            for k in range(nnzRow):
                # Get column index
                jj = TComRow[k]
                # The problem is symetric
                if jj != ii:
                    # Copy partition to workspace
                    TWork_lil = TCom_lil.copy()
                    uWork_lil = uCom_lil.copy()
                    memberWork = member.copy()
                    
                    # Add node jj to community ii
                    TWork_lil[ii] = (uWork_lil[ii, 0] / (uWork_lil[ii, 0] + uWork_lil[jj, 0])) * TWork_lil[ii] + (uWork_lil[jj, 0] / (uWork_lil[ii, 0] + uWork_lil[jj, 0])) * TWork_lil[jj]
                    TWork_lil[:, ii] = TWork_lil[:, ii] + TWork_lil[:, jj]
                    uWork_lil[ii] += uWork_lil[jj]
                    
                    # Remove Node 
                    TWork_lil[jj] = 0
                    TWork_lil[:, jj] = 0
                    uWork_lil[jj] = 0
                    memberWork[memberWork == jj] = ii
                    

                    # Get codelength
                    (codelengthWork, crap, crap, crap) = twolevelCodelengthFromTrans(T_csr, memberWork, u=u_lil, Hu=Hu, uMod=uWork_lil, TMod=TWork_lil)
                    
                    if codelengthWork < codeMin:
                        codeMin = codelengthWork
                        argCodeMin = (ii, jj)
                        
        # Apply best agglomeration
        (ii, jj) = argCodeMin
        TCom_lil[ii] = (1 - uCom_lil[jj, 0]) * TCom_lil[ii] + uCom_lil[jj, 0] * TCom_lil[jj]
        TCom_lil[jj] = 0
        TCom_lil[:, ii] = TCom_lil[:, ii] + TCom_lil[:, jj]
        TCom_lil[:, jj] = 0
        uCom_lil[ii] += uCom_lil[jj]
        uCom_lil[jj] = 0
        member[member == jj] = ii
        codelength.append(codeMin)
        nMod = uCom_lil.nnz
        (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u_lil, Hu=Hu, uMod=uCom_lil, TMod=TCom_lil)
 
        print 'Moving %d to %d...' % (jj, ii)
        print 'Codelength after iteration %d is %f with %d modules.' % (nIter, codeMin, nMod)
        print HQ
        print hQM
        print member
        nIter += 1
   
    return codelength


def greedyCodelength2(T):
    if not sparse.issparse(T):
        print 'Converting matrix to LIL...'
        T_lil = sparse.lil_matrix(T)
    elif T.format != 'lil':
        print 'Converting matrix to LIL...'
        T_lil = sparse.lil_matrix(T)
    else:
        T_lil = T

    T_csr = T_lil.tocsr()

    # Initialization
    print 'Initializing...'
    N = T_lil.shape[0]

    # Initialize membership vector
    member = np.arange(N)
    membership = np.zeros((N, N), dtype=int)

    # Get stationnary distribution and its entropy
    (crap, u) = atgraph.arnoldi(T_lil, k=1)
    u = np.abs(u).T
    u = np.matrix(u / u.sum()).reshape(u.shape[0], 1)
    Hu = atmath.entropy(u)
    print 'Initial entropy of stationnary distribution = %f' % Hu

    # Get initial codelength (the entropy of the Markov process)
    logT_csr = T_csr.copy()
    logT_csr.data = logT_csr.data * np.log2(logT_csr.data)
    hM = (-u.T * logT_csr.sum(1))[0, 0]
    print 'Initial codelength or entropy rate is %f' % hM

    modIndex = np.unique(member)
    nMod = modIndex.shape[0]

    TCom_lil = T_lil.copy()
    uCom = np.array(u)

    (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u, Hu=Hu, uMod=u, TMod=T_csr)
    
    print HQ
    print hQM
    print L

    # Loop on the number of iteration
    codelength = [hM] 
    nIter = 1
    while nMod > 1:
        # Search
        codeMin = Hu * 100
        argCodeMin = (0, 0)
        TCom_csr = TCom_lil.tocsr()
        TCom_csc = TCom_lil.tocsc()
        for ii in range(N):
            nnzRow = TCom_csr[ii].nnz
            TComRow = TCom_csr[ii].indices
            print ii
            for k in range(nnzRow):
                # Get column index
                jj = TComRow[k]
                # The problem is symetric
                if jj > ii:
                    stat = 0.
                    dyn1 = 0.
                    dyn1a = 0.
                    dyn1b = 0.
                    dyn2 = 0.
                    dyn3 = 0.
                    dyn3a = 0.
                    dyn3b = 0.
                    # Contribution of stationary distribution of partion
                    stat = uCom[ii, 0] * np.log2(uCom[ii, 0]) + uCom[jj, 0] * np.log2(uCom[jj, 0]) - (uCom[ii, 0] + uCom[jj, 0]) * np.log2(uCom[ii, 0] + uCom[jj, 0])
                    # Contribution of the dynamics on the partition
                    dyn1 = - ((uCom[ii, 0] + uCom[jj, 0]) * ((uCom[ii, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[ii] + uCom[jj, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[jj]).data[0] * np.log2((uCom[ii, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[ii] + uCom[jj, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[jj]).data[0]))).sum()
#                    dyn1a = uCom[ii, 0] * (TCom_csr[ii].data * np.log2(TCom_csr[ii].data)).sum()
#                    dyn1b = uCom[jj, 0] * (TCom_csr[jj].data * np.log2(TCom_csr[jj].data)).sum()

                    if TCom_csr[ii, ii] + TCom_csr[jj, jj] > 0.:
                        dyn2 = - ((uCom[ii, 0] + uCom[jj, 0]) * (TCom_csr[ii, ii] + TCom_csr[jj, jj]) * np.log2(TCom_csr[ii, ii] + TCom_csr[jj, jj])).sum()
                    else:
                        dyn2 = 0.
                        
                    dyn3 = - (uCom[(TCom_csc[:, ii] + TCom_csc[:, jj]).indices, 0] * (TCom_csc[:, jj] + TCom_csc[:, ii]).data * np.log2((TCom_csc[:, jj] + TCom_csc[:, ii]).data)).sum()
#                    dyn3a = (uCom[TCom_csc[:, jj].indices, 0] * TCom_csc[:, jj].data * np.log2(TCom_csc[:, jj].data)).sum()
#                    dyn3b = (uCom[TCom_csc[:, ii].indices, 0] * TCom_csc[:, ii].data * np.log2(TCom_csc[:, ii].data)).sum()
                    
                    dyn = dyn1 + dyn1a + dyn1b + dyn2 + dyn3 + dyn3a + dyn3b

                    # Update codelength
                    codelengthWork = codelength[-1] - stat + dyn
                    
                    
                    if codelengthWork < codeMin:
                        statMin = stat
                        dynMin = dyn
                        codeMin = codelengthWork
                        argCodeMin = (ii, jj)
                        
        # Apply best agglomeration
        (ii, jj) = argCodeMin
        TCom_lil[ii] = uCom[ii, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[ii] + uCom[jj, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[jj]
        TCom_lil[jj] = 0
        TCom_lil[:, ii] = TCom_lil[:, ii] + TCom_lil[:, jj]
        TCom_lil[:, jj] = 0
        uCom[ii] += uCom[jj]
        uCom[jj] = 0
        member[member == jj] = ii
        codelength.append(codeMin)
        nMod = np.sum(uCom > 0)
        (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u, Hu=Hu, uMod=uCom, TMod=TCom_lil)
 
        print 'Moving %d to %d...' % (jj, ii)
        print 'Codelength after iteration %d is %f with %d modules.' % (nIter, L, nMod)
        print 'H(Q) = ', HQ
        print 'h(q) = ', hQM
        print HQ - stat
        print hQM - dyn

        membership[nIter - 1] = member
        print member

        nIter += 1
   
    return (membership, codelength)


# def greedyCodelength3(T):
#     if not sparse.issparse(T):
#         print 'Converting matrix to LIL...'
#         T_lil = sparse.lil_matrix(T)
#     elif T.format != 'lil':
#         print 'Converting matrix to LIL...'
#         T_lil = sparse.lil_matrix(T)
#     else:
#         T_lil = T

#     T_csr = T_lil.tocsr()

#     # Initialization
#     print 'Initializing...'
#     N = T_lil.shape[0]

#     # Initialize membership vector
#     member = np.arange(N)
#     membership = np.zeros((N, N), dtype=int)

#     # Get stationnary distribution and its entropy
#     (crap, u) = atgraph.arnoldi(T_lil, k=1)
#     u = np.abs(u).T
#     u = np.matrix(u / u.sum()).reshape(u.shape[0], 1)
#     Hu = atmath.entropy(u)
#     print 'Initial entropy of stationnary distribution = %f' % Hu

#     # Get initial codelength (the entropy of the Markov process)
#     logT_csr = T_csr.copy()
#     logT_csr.data = logT_csr.data * np.log2(logT_csr.data)
#     hM = (-u.T * logT_csr.sum(1))[0, 0]
#     print 'Initial codelength or entropy rate is %f' % hM

#     modIndex = np.unique(member)
#     nMod = modIndex.shape[0]

#     TCom_lil = T_lil.copy()
#     uCom = np.array(u.copy())

#     (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u, Hu=Hu, uMod=u, TMod=T_csr)
    
#     print HQ
#     print hQM
#     print L

#     # Loop on the number of iteration
#     codelength = [hM] 
#     statlength = [HQ]
#     nIter = 1
#     while nMod > 1:
#         # Search
#         statMax = 0.
#         argStatMax = (0, 0)
#         TCom_csr = TCom_lil.tocsr()
#         TCom_csc = TCom_lil.tocsc()
#         for ii in range(N):
#             nnzRow = TCom_csr[ii].nnz
#             TComRow = TCom_csr[ii].indices
#             for k in range(nnzRow):
#                 # Get column index
#                 jj = TComRow[k]
#                 # The problem is symetric
#                 if jj > ii:
#                     # Contribution of stationary distribution of partion
#                     stat = uCom[ii, 0] * np.log2(uCom[ii, 0]) + uCom[jj, 0] * np.log2(uCom[jj, 0]) - (uCom[ii, 0] + uCom[jj, 0]) * np.log2(uCom[ii, 0] + uCom[jj, 0])
#                     # Update codelength
#                     statWork = statlength[-1] + stat
                    
#                     if statWork > statMax:
#                         statMax = statWork
#                         argStatMax = (ii, jj)
                        
#         # Apply best agglomeration
#         (ii, jj) = argStatMax
#         TCom_lil[ii] = uCom[ii, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[ii] + uCom[jj, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[jj]
#         TCom_lil[jj] = 0
#         TCom_lil[:, ii] = TCom_lil[:, ii] + TCom_lil[:, jj]
#         TCom_lil[:, jj] = 0
#         uCom[ii] += uCom[jj]
#         uCom[jj] = 0
#         member[member == jj] = ii
#         nMod = np.sum(uCom > 0)
#         (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u, Hu=Hu, uMod=uCom, TMod=TCom_lil)
#         codelength.append(L)
#         statlength.append(statMax)
 
#         print 'Moving %d to %d...' % (jj, ii)
#         print 'Codelength after iteration %d is %f with %d modules.' % (nIter, L, nMod)
#         print 'H(Q) = ', HQ
#         print 'h(q) = ', hQM
        
#         membership[nIter - 1] = member
#         nIter += 1
   
#     return (membership, codelength, statlength)

# # def greedyCodelength4(T):
# #     if not sparse.issparse(T):
# #         print 'Converting matrix to LIL...'
# #         T_lil = sparse.lil_matrix(T)
# #     elif T.format != 'lil':
# #         print 'Converting matrix to LIL...'
# #         T_lil = sparse.lil_matrix(T)
# #     else:
# #         T_lil = T

# #     T_csr = T_lil.tocsr()

# #     # Initialization
# #     print 'Initializing...'
# #     N = T_lil.shape[0]

# #     # Initialize membership vector
# #     member = np.arange(N)
# #     membership = np.zeros((N, N), dtype=int)

# #     # Get stationnary distribution and its entropy
# #     (crap, u) = atgraph.arnoldi(T_lil, k=1)
# #     u = np.abs(u).T
# #     u = np.matrix(u / u.sum()).reshape(u.shape[0], 1)
# #     Hu = atmath.entropy(u)
# #     print 'Initial entropy of stationnary distribution = %f' % Hu

# #     # Get initial codelength (the entropy of the Markov process)
# #     logT_csr = T_csr.copy()
# #     logT_csr.data = logT_csr.data * np.log2(logT_csr.data)
# #     hM = (-u.T * logT_csr.sum(1))[0, 0]
# #     print 'Initial codelength or entropy rate is %f' % hM

# #     modIndex = np.unique(member)
# #     nMod = modIndex.shape[0]

# #     TCom_lil = T_lil.copy()
# #     uCom = np.array(u.copy())

# #     (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u, Hu=Hu, uMod=u, TMod=T_csr)
    
# #     print HQ
# #     print hQM
# #     print L

# #     # Loop on the number of iteration
# #     codelength = [hM] 
# #     statlength = [HQ]
# #     nIter = 1
# #     while nMod > 1:
# #         # Search
# #         statMax = 0.
# #         argStatMax = (0, 0)
# #         TCom_csr = TCom_lil.tocsr()
# #         TCom_csc = TCom_lil.tocsc()
# #         Qr = np.tile(uCom, (1, N))
# #         Qs = np.tile(uCom.T, (N, 1))
# #         stat = np.triu(-np.multiply(Qr + Qs, np.log2(Qr + Qs)) + np.multiply(Qr, np.log2(Qr)) + np.multiply(Qs, np.log2(Qs)), k=1) - np.tril(np.ones((N, N)), k=0)
# #         argStatMax = np.unravel_index(np.nanargmax(stat), stat.shape)
# #         print stat
# #         print argStatMax
# #         statMax = stat[argStatMax[0], argStatMax[1]]
                        
# #         # Apply best agglomeration
# #         (ii, jj) = argStatMax
# #         TCom_lil[ii] = uCom[ii, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[ii] + uCom[jj, 0] / (uCom[ii, 0] + uCom[jj, 0]) * TCom_lil[jj]
# #         TCom_lil[jj] = 0
# #         TCom_lil[:, ii] = TCom_lil[:, ii] + TCom_lil[:, jj]
# #         TCom_lil[:, jj] = 0
# #         uCom[ii] += uCom[jj]
# #         uCom[jj] = 0
# #         member[member == jj] = ii
# #         nMod = np.sum(uCom > 0)
# #         (L, Hu, HQ, hQM) = twolevelCodelengthFromTrans(T_csr, member, u=u, Hu=Hu, uMod=uCom, TMod=TCom_lil)
# #         codelength.append(L)
# #         statlength.append(statMax)
 
# #         print 'Moving %d to %d...' % (jj, ii)
# #         print 'Codelength after iteration %d is %f with %d modules.' % (nIter, L, nMod)
# #         print 'H(Q) = ', HQ
# #         print 'h(q) = ', hQM
        
# #         membership[nIter - 1] = member
# #         nIter += 1
   
# #     return (membership, codelength, statlength)

