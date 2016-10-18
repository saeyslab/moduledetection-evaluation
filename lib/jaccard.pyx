cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, log, erf

@cython.cdivision(True)
#@cython.boundscheck(False)
cpdef cal_similaritymatrix_jaccard(np.ndarray[np.uint8_t, ndim=2] X, np.ndarray[np.uint8_t, ndim=2] Y):
    """
    Calculates the setmatrix given two module matrices (genes in columns, modules/clusters in rows) using the jaccard
    """
    
    cdef int i
    cdef int j
    cdef int k
    
    cdef double teller = 0
    cdef double noemer = 0

    cdef int nG = X.shape[1]
    cdef int nMx = X.shape[0]
    cdef int nMy = Y.shape[0]
    
    cdef np.ndarray[np.float64_t, ndim=2] S = np.zeros((nMx, nMy), dtype=np.float64)
    
    for i in range(nMx):
        for j in range(nMy):
            # calculate jaccard   
            teller = 0
            noemer = 0
            for k in range(nG):
                if X[i,k] or Y[j,k]:
                    noemer += 1
                    if X[i,k] and Y[j,k]:
                        teller += 1
                        
            S[i,j] = teller/noemer
            
    return S