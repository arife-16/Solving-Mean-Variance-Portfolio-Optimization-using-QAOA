
# qaoa_maxcut_energy.pyx
cimport numpy as np
cimport cython

cdef extern from *:
    """
    #include <math.h>
    double compute_maxcut_energy_c(int state, int* edge_u, int* edge_v, double* weights, int n_edges) {
        double energy = 0.0;
        for (int i = 0; i < n_edges; i++) {
            int bit_u = (state >> edge_u[i]) & 1;
            int bit_v = (state >> edge_v[i]) & 1;
            energy += (bit_u ^ bit_v) * weights[i];
        }
        return energy;
    }
    """
    double compute_maxcut_energy_c(int, int*, int*, double*, int)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_maxcut_energy_cython(int state, np.ndarray[np.int32_t, ndim=1] edge_u, 
                                 np.ndarray[np.int32_t, ndim=1] edge_v, 
                                 np.ndarray[np.float64_t, ndim=1] weights) -> float:
    return compute_maxcut_energy_c(state, &edge_u[0], &edge_v[0], &weights[0], len(edge_u))
