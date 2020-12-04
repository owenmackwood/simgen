from typing import Union, Tuple, Type
import numpy as np
from numba import njit
from collections import namedtuple

dtype = np.float32

mmap_array = namedtuple('mmap_array', ['identifier', 'dtype', 'shape', 'filename', 'T'])


def allocate_aligned(shape: Union[int, Tuple[int, ...]],
                     alignment: int = 32, dtype: Type[np.number] = np.float32) -> np.ndarray:
    if alignment <= 32:
        a = numba_allocate_aligned(shape, dtype)
    else:
        print("WARNING: Allocated with numpy!")
        size = np.prod(shape)
        dtype = np.dtype(dtype)
        nbytes = size * dtype.itemsize
        buf = np.zeros(nbytes + alignment, dtype=np.uint8)
        start_index = -buf.ctypes.data % alignment
        a = buf[start_index:start_index + nbytes].view(dtype).reshape(shape)
    assert not a.ctypes.data % alignment
    return a


@njit(cache=False)
def numba_allocate_aligned(
        shape: Union[int, Tuple[int, ...]], dtype: Type[np.number]
) -> np.ndarray:
    # NUMBA allocates arrays with 32-byte alignment
    return np.zeros(shape, dtype=dtype)


def reallocate_aligned(a: np.ndarray, alignment: int = 32) -> np.ndarray:
    if alignment <= 32:
        b = numba_allocate_aligned(a.shape, a.dtype)
    else:
        b = allocate_aligned(a.shape, alignment, a.dtype)
    np.copyto(b, a)
    a = b
    assert not a.ctypes.data % alignment
    return a


connection_sparse = namedtuple('connection_sparse', ['csr_indices', 'row_ind', 'col_ind', 'conn_prob'])
csr_indices = namedtuple('csr_indices', ['indptr', 'indices', 'data'])


def compute_connections_sparse(
        connection_distance, p, N_pre, N_post, zero_diagonal, circular_connectivity, in_degree_cv = None,
):
    """
    where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
    """
    conns, coo, cp = _compute_connections_base(connection_distance, p, N_pre, N_post,
                                               zero_diagonal, circular_connectivity, in_degree_cv)

    row_ind = reallocate_aligned(coo.row)
    col_ind = reallocate_aligned(coo.col)

    csr = coo.tocsr()
    indptr = reallocate_aligned(csr.indptr)
    indices = reallocate_aligned(csr.indices)
    data = reallocate_aligned(csr.data)

    return connection_sparse(csr_indices(indptr, indices, data), row_ind, col_ind, cp)


def _compute_connections_base(
        connection_distance, p, N_pre, N_post, zero_diagonal, circular_connectivity, in_degree_cv
):
    from scipy.sparse import coo_matrix
    if circular_connectivity:
        pi_scale = 1. / (np.pi * connection_distance) ** 2
    else:
        pi_scale = connection_distance * np.pi
    d_min, d_max = -np.pi, np.pi
    pos_pre = np.linspace(d_min, d_max, N_pre, endpoint=False)
    pos_post = np.linspace(d_min, d_max, N_post, endpoint=False)

    fixed_in = in_degree_cv == 0.

    k = p*(N_pre - 1) if zero_diagonal else p*N_pre
    if in_degree_cv is None or in_degree_cv < 0.:
        sd_k = np.sqrt(k)
        in_degree_cv = sd_k / k
        print('CV of in-degree not specified, using CV {:1.2f} for K {:.1f}'.format(in_degree_cv, k))

    if not fixed_in:
        sd_k = in_degree_cv * k
        c = np.random.normal(loc=k, scale=sd_k, size=N_post)
        c = np.round(c).astype(np.int)
        c_zero = c <= 0
        print('Number of zero in-degrees', c_zero.sum())
        c[c_zero] = 1
    else:
        c0 = int(np.round(k))
        print('Fixed in-degree of {}'.format(c0))
        c = np.ones(N_post, dtype=np.int) * max(1, c0)
    use_choice = 1  # Don't change this unless you're REALLY sure.

    conns = np.zeros((N_post, N_pre), dtype=dtype)
    cp = np.zeros((N_post, N_pre), dtype=dtype)
    retried = 0
    for i in range(N_post):
        d_curr = pos_post[i] - pos_pre
        d_curr[d_curr < d_min] -= 2*d_min
        d_curr[d_curr > d_max] -= 2*d_max
        if connection_distance == np.infty:
            ps = np.ones_like(d_curr)
        elif circular_connectivity:
            ps = np.exp(pi_scale * (np.cos(d_curr) - 1))
            # ps = vonmises.pdf(d_curr, pi_scale)
        else:
            ps = np.exp(-np.abs(d_curr)/pi_scale)
        if zero_diagonal:
            ps[np.abs(d_curr) < 1e-12] = 0
        ps /= ps.sum()
        # ps += f_global / ps.size
        nnz = ps.nonzero()[0].size
        if nnz < c[i]:
            print('WARNING: Fewer pre-synaptic cells with non-zero connection '
                  'probability than desired in-degree! {} {}'.format(nnz, c[i]))
        c_i = min(nnz, c[i])
        cp[i, :] = ps * c_i
        if use_choice:
            # Seems to under-sample near areas of high connection density?
            j = np.random.choice(N_pre, size=c_i, replace=False, p=ps)
        else:
            # This is worse than random.choice, does not produce the required in-degree c_i
            ps *= c_i
            retry = True
            while retry:
                s = np.random.uniform(size=N_pre)
                j = np.argwhere(ps > s)
                # retry = j.size < c_i
                retry = j.size == 0
                if retry:
                    retried += 1
        # if fixed_in:
        #     j = np.random.choice(N_pre, size=c_i, replace=False, p=ps)
        # else:
        #     ps *= c_i
        #     retry = True
        #     while retry:
        #         s = np.random.uniform(size=N_pre)
        #         j = np.argwhere(ps > s)
        #         retry = j.size == 0
        #         if retry:
        #             retried += 1
        conns[i, j] = 1.

    # if PLOT_CONNECTIONS:
    #     plot_connection(scale, p, f_global, d_pre, N_pre, d_post, N_post, zero_diagonal, conns)

    # j_rescale = scale_j(conns, p, N_pre, J, J_scale, retried)
    coo = coo_matrix(conns)
    # coo.data *= j_rescale
    in_w = coo.sum(axis=1)
    in_mu = in_w.mean()
    in_std = in_w.std()
    print('Input weight CV {}, mean {}, std {}'.format(in_std/in_mu, in_mu, in_std))
    # if J_var > 0.:
    #     print('Before randomization')
    #     report_connection(conns, p, N_pre, retried)
    #     j = np.random.normal(loc=j_rescale, scale=np.abs(j_rescale*J_var), size=coo.data.size)
    #     # lt0 = j < 0.
    #     # print("Produced {} < 0".format(lt0.sum()/j.size))
    #     # j[lt0] = 0.
    #     coo.data[:] = j
    #     print('After randomization')
    #     report_connection(coo.toarray(), p, N_pre, retried)
    conns = coo.toarray()

    return conns, coo, cp


# def scale_j(conns, p, N_pre, J, J_scale, retried):
#     calc_mu, calc_std, calc_var, mu_scale, std_scale, var_scale = report_connection(conns, p, N_pre, retried)
#     if J_scale == 'mu':
#         J_calc, J_emp = J / calc_mu, J / mu_scale
#         print("J/p*N: {} J/mu: {} diff*N_pre/J: {}".format(J_calc, J_emp, (J_calc-J_emp)*N_pre/J))
#         j_rescale = J / mu_scale
#     elif J_scale == 'std':
#         J_calc, J_emp = J/calc_std, J/std_scale
#         print("J/sqrt(p*(1-p)*N): {} J/std: {} diff*N_pre/J: {}".format(J_calc, J_emp, (J_calc-J_emp)*N_pre/J))
#         j_rescale = J / std_scale
#     elif J_scale == 'var':
#         J_calc, J_emp = J/calc_var, J/var_scale
#         print("J/p*(1-p)*N: {} J/var: {} diff*N_pre/J: {}".format(J_calc, J_emp, (J_calc-J_emp)*N_pre/J))
#         j_rescale = J / var_scale
#     else:
#         j_rescale = J
#     return j_rescale
