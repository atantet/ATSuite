import numpy as np
from scipy import sparse

def sparse2Compressed(fname, spmat, fmt='%.18e'):
    nnz = spmat.nnz
    (M, N) = spmat.shape
    typeName = spmat.format

    # Open destination file
    f = open(fname, 'w')

    # Write matrix dimension
    f.write('%s\t%d\t%d\t%d\n' % (typeName, M, N, nnz))
    
    # Write data list
    for nz in range(nnz):
        f.write((fmt % spmat.data[nz]) + '\t')
    f.write('\n')

    # Write column indices list
    for nz in range(nnz):
        f.write(('%d' % spmat.indices[nz]) + '\t')
    f.write('\n')

    # Write row pointers list
    for ii in np.arange(spmat.indptr.shape[0]):
        f.write(('%d' % spmat.indptr[ii]) + '\t')
    f.write('\n')
    
    f.close()

def compressed2Sparse(fname, dtype=float):
    
    # Open destination file
    f = open(fname, 'r')

    # Read matrix dimension and number of elements
    buf = f.readline().split()
    (matType, M, N, nnz) = (buf[0], int(buf[1]), int(buf[2]), int(buf[3]))

    # Read data list
    buf = f.readline().split()
    data = [dtype(buf[nz]) for nz in range(nnz)]

    # Read column indices list
    buf = f.readline().split()
    indices = [int(buf[nz]) for nz in range(nnz)]

    # Read row pointers list
    buf = f.readline().split()
    indptr = [int(buf[ii]) for ii in np.arange(len(buf))]

    # Create matrix
    if matType.lower() == 'csr':
        spmat = sparse.csr_matrix((data, indices, indptr), (M, N),
                                  dtype=dtype)
    elif matType.lower() == 'csc':
        spmat = sparse.csc_matrix((data, indices, indptr), (M, N),
                                  dtype=dtype)
    
    f.close()

    return spmat
