cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, log, erf

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef tuple cal_ebcubed(np.ndarray[np.uint8_t, ndim=2] Amembership, np.ndarray[np.uint8_t, ndim=2] Bmembership, np.ndarray[np.float64_t, ndim=2] jaccards):
    cdef int nG = Amembership.shape[0]
    cdef int nB = Bmembership.shape[1]
    cdef int nM = Amembership.shape[1]

    cdef int Ashared
    cdef int Bshared

    cdef double subprecision
    cdef double subrecall

    cdef double subprecision_denominator
    cdef double subrecall_denominator

    cdef double Aphi
    cdef double Bphi

    cdef double bestsim = 0

    cdef double Aulesizeweight = 0

    cdef np.ndarray[np.float64_t, ndim=1] recalls = np.zeros(nG, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] precisions = np.zeros(nG, dtype=np.float64)

    for i in range(nG):
        subprecision = 0
        subrecall = 0

        subprecision_denominator = 0
        subrecall_denominator = 0

        for j in range(nG):
            Bshared = 0
            Aphi = 0
            for k in range(nB):
                if Bmembership[i,k] & Bmembership[j,k]:
                    # calculate shared
                    Bshared += 1

                    # find most similar module
                    bestsim = 0
                    for l in range(nM):
                        if jaccards[k, l] > bestsim:
                            bestsim = jaccards[k, l]
                    Aphi += bestsim

            Ashared = 0
            Bphi = 0
            for k in range(nM):
                if Amembership[i,k] & Amembership[j,k]:
                    # calculate shared
                    Ashared += 1

                    # find most similar module
                    bestsim = 0
                    for l in range(nB):
                        if jaccards[l, k] > bestsim:
                            bestsim = jaccards[l,k]
                    Bphi += bestsim

            if Bshared > 0:
                subprecision += Aphi/<float>Bshared * min(Ashared, Bshared)/<float>Bshared # division of two integers -> float
                subprecision_denominator += 1
            if Ashared > 0:
                subrecall += Bphi/<float>Ashared * min(Ashared, Bshared)/<float>Ashared
                subrecall_denominator += 1

        if subprecision > 0:
            precisions[i] = subprecision/subprecision_denominator

        if subrecall > 0:
            recalls[i] = subrecall/subrecall_denominator

    return recalls, precisions
