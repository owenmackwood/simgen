# from six import iteritems
# from functools import partial
# from scipy.sparse.sparsetools import csr_matvec
# from scipy.sparse import csr_matrix
# import numexpr as ne
# from collections import namedtuple
from typing import NamedTuple, List, Tuple, Dict, Union
import numpy as np
from utils import dtype
from enum import IntEnum


class InterneuronPlasticity(IntEnum):
    NONE = 0
    HEBBIAN = 1
    NONHEBB = 2
    ANTIHEBB = 3
    EXC2INH = 4
    INH_GAIN = 5
    INH_THRESH = 6
    BCM = 7
    AFF2INH = 8


class PlasticitySubtype(IntEnum):
    LOCAL = 1
    DIFFUSIVE = 0
    ANTEROGRADE = -1
    RETROGRADE = -2
    FULL_GRADIENT = -3
    BCM = -4


c_library = NamedTuple('c_library', [('h_file_path', str), ('so_file_path', str),
                                     ('p_types_names', List[Tuple[str, str]]), ('h_c_source', str)])
weight_updater = NamedTuple('weight_updater',
                            [('plasticity_init', str), ('update_weights', str), ('update_functions', str)])

function_definitions = NamedTuple('function_definitions',
                                  [('main_declaration', str), ('main_definition', str),
                                   ('p_types_names', List[Tuple[str, str]])])


def _prepare_inner_c(n_e: int, n_i: int, e2i_nnz: int, i2e_nnz: int, a2i_nnz: int,
                     alpha_r: float, alpha_p: float, sp_type: str, sp_trans: float,
                     plastic_var: str, weight_update_on_steps: bool, i2e_binary: bool, n_plastic_str: str,
                     has_k: bool, i2e_plastic: bool, calibrate_syn: str,
                     do_ou_exc: bool, do_ou_inh: bool, do_aff_inh: bool, n_a2i: int,
                     has_e2e: bool, has_i2i: bool,
                     build_path: str, do_print_arrays: bool, rec_all_plastic: bool, do_mmap: bool) -> c_library:
    import subprocess as sp
    import os
    # from distutils.core import setup
    # from distutils.extension import Extension
    # from Cython.Distutils import build_ext
    # import snep.library.rates as slr
    # global h_file_path, so_file_path
    use_sigmoid = False
    do_pyx = 0
    # build_path = build_path or slr.__path__[0]

    c_file_path = os.path.join(build_path, "inner_impl.c")
    h_file_path = os.path.join(build_path, "inner_impl.h")
    if do_pyx:
        so_file_path = os.path.join(build_path, "inner_impl.o")
    else:
        so_file_path = os.path.join(build_path, "inner_impl.so")

    mkl_threads = 1
    dft = f'''
#define N_REAL NE
#define N_CMPLX (N_REAL/2+1)
MKL_Complex8 in_out_cmplx[N_CMPLX], k_cmplx[N_CMPLX];
DFTI_DESCRIPTOR_HANDLE init_dfti(const float *k)
{{
    DFTI_DESCRIPTOR_HANDLE hand = 0;
    MKL_LONG status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG)N_REAL);
    if (0 == status) status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (0 == status) status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (0 == status) status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, 1. / N_REAL);
    //if (0 == status) status = DftiSetValue(hand, DFTI_THREAD_LIMIT, {mkl_threads});
    if (0 == status) status = DftiCommitDescriptor(hand);
    if (0 != status) printf("ERROR, status = %ld \\n", status);
    DftiComputeForward(hand, (void *)k, k_cmplx);
    return hand;
}}

void spatial_filter(DFTI_DESCRIPTOR_HANDLE hand, float* r_exc, float* r_hat)
{{
    DftiComputeForward(hand, r_exc, in_out_cmplx);
    vcMul(N_CMPLX, in_out_cmplx, k_cmplx, in_out_cmplx);
    DftiComputeBackward(hand, in_out_cmplx, r_hat);
}}
    ''' if has_k else ''

    non_linearity = 'typedef enum {GAIN, THRESH, OTHER} derivative_wrt;'

    src_dir = os.path.join(os.path.dirname(__file__), 'src')

    if alpha_r < np.infty or alpha_p < np.infty:
        with open(os.path.join(src_dir, 'soft_plus_with_max.c')) as f:
            non_linearity += f.read()
    if not (alpha_r < np.infty and alpha_p < np.infty):
        with open(os.path.join(src_dir, 'piece_wise.c')) as f:
            non_linearity += f.read()
    if alpha_r < np.infty:
        r_nl = 'soft_plus_with_max'
        r_nld = 'soft_plus_with_max_derivative'
    else:
        r_nl = 'piece_wise'
        r_nld = 'piece_wise_derivative'
    if alpha_p < np.infty:
        p_nl = 'soft_plus_with_max'
        p_nld = 'soft_plus_with_max_derivative'
    else:
        p_nl = 'piece_wise'
        p_nld = 'piece_wise_derivative'

    non_linearity += f'''
void plastic_nonlinearity(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y)
{{
    {p_nl}(max_y, alpha, n, x, y, NULL, NULL);
}}                               
void plastic_nonlinearity_deriv(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y)
{{
    {p_nld}(max_y, alpha, n, x, y, NULL, NULL, OTHER);
}} 
void rate_nonlinearity(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain)
{{
    {r_nl}(max_y, alpha, n, x, y, thresh, gain);
}}                               
void rate_nonlinearity_deriv(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain, const derivative_wrt wrt)
{{
    {r_nld}(max_y, alpha, n, x, y, thresh, gain, wrt);
}}                               
'''

    normalize_sparse = ''
    with open(os.path.join(src_dir, 'norm_sparse.c')) as f:
        normalize_sparse += f.read()

    c_includes = '''
// Need the following define for madvise
#define _GNU_SOURCE

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <assert.h>

//////////////////////////
//// Includes for mmap ///
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
//////////////////////////

#include "inner_impl.h" 
#include "mkl.h" // Must include after typedef for MKL_INT
'''

    # float_repr = '.6a '
    def_print_arrays = '#define PRINT_ARRAYS' if do_print_arrays else ''
    align = 64

    array_print_functions = ''
    if do_print_arrays:
        with open(os.path.join(src_dir, 'print_array.c')) as f:
            array_print_functions += f.read()

    c_file_contents = f'''
{c_includes}
#define NE {n_e}
#define NI {n_i}
#define E2I_NNZ {e2i_nnz}
#define I2E_NNZ {i2e_nnz}
#define A2I_NNZ {a2i_nnz}
#define N_PLASTIC {n_plastic_str}
#define N_A2I {n_a2i}
#define ALIGNMENT {align}
const char *no_trans = "N", *do_trans = "T";

{dft}

{def_print_arrays}
#ifdef PRINT_ARRAYS
    #define pe2(s, a, m, n) print_2d_float(s, a, m, n)
    #define pia(s, a, n) print_1d_int(s, a, n)
    #define psa(s, a, nnz, m, n, columns, row_ptr) print_sparse(s, a, nnz, m, n, columns, row_ptr)
#else
    #define pe2(s, a, m, n)
    #define pia(s, a, n)
    #define psa(s, a, nnz, m, n, columns, row_ptr)
#endif
{array_print_functions}

{normalize_sparse}

{non_linearity}'''

    h_file_contents = '''
typedef signed int MKL_INT;
typedef unsigned int MKL_UINT;
'''
    function_definitions = _inner_fn(sp_type, sp_trans, plastic_var, weight_update_on_steps, i2e_binary,
                                     has_k, i2e_plastic, calibrate_syn,
                                     do_ou_exc, do_ou_inh, do_aff_inh,
                                     has_e2e, has_i2i,
                                     do_print_arrays, rec_all_plastic, do_mmap)
    h_file_contents += function_definitions.main_declaration + ';'
    c_file_contents += function_definitions.main_definition

    with open(c_file_path, mode='w') as f:
        f.write(c_file_contents)
    with open(h_file_path, mode='w') as f:
        f.write(h_file_contents)

    # ld_library_path = os.path.expanduser('~/anaconda/lib:{}:'.format(build_path))
    # os.environ["LD_LIBRARY_PATH"] = ld_library_path
    cmd = 'gcc -march=native -std=c99 -msse4 -mavx -O3 -fPIC -m64 ' \
          f'-Wdouble-promotion -Wshadow -Wall -Wextra -Wl,--no-as-needed "{c_file_path}"'

    # -fopt-info-all : outputs optimization information
    #  -Wrestrict : warns when aliased parameters passed to restrict arguments (GCC 7)
    extra_incs = [' -I' + os.path.expanduser(s) for s in ["~/intel/mkl/include"]]
    cmd += ' '.join(extra_incs)
    user_path = os.path.expanduser('~')
    ll_dirs = [f' -L{user_path}/{s}' for s in ['intel/mkl/lib/intel64',
                                               'intel/lib/intel64']]  # [' -L{0}/anaconda/lib'.format(os.path.expanduser('~'))]
    cmd += ' '.join(ll_dirs)
    ll_files = [' -l' + l for l in ['mkl_intel_lp64', 'mkl_core', 'iomp5']]  # 'mkl_gf_lp64', 'mkl_gnu_thread',
    cmd += ' '.join(ll_files)
    if do_pyx:
        cmd += ' -c'
    else:
        cmd += ' -shared'
    cmd += f'  -o "{so_file_path}"'
    print(f'COMPILING {sp_type} {sp_trans} iSP:{i2e_plastic}')
    print(cmd)
    try:
        sp.run(cmd, check=True, shell=True, capture_output=True)
    except sp.CalledProcessError as e:
        if b'#include "mkl.h"' in e.stderr:
            print(80 * "=" + "\nYou need to have Intel MKL installed to compile the simulator.\n" + 80 * "=")
            exit()

    return c_library(h_file_path, so_file_path, function_definitions.p_types_names,
                     h_file_contents + c_file_contents)


def _local_plasticity(sp_type: str) -> Tuple[str, str]:
    p_types_names = [('const float', 'eta'),
                     ('const float', 'j_max'),
                     ('float *', 'error'),
                     ('const float', 'alpha_p'),
                     ('float *', 'plastic_data'),
                     ('float *', 'plastic_delta'),
                     ('float *', 'e2i_data_pos'),
                     ('const MKL_INT *', 'e2i_data_j')]
    plasticity_local_floats = ['plastic_logistic[E2I_NNZ]', ]
    plasticity_local_init = ''

    if sp_type != 'nonhebb':
        p_types_names += [('const float *', 'r_inh'),
                          ('const MKL_INT *', 'e2i_data_i')]
        plasticity_local_floats += ['r_inh_i[E2I_NNZ]']

    if sp_type == 'hebbian':
        post_rates = '''
        vsPackV(E2I_NNZ, r_inh, e2i_data_i, r_inh_i); // Take equiv.
        vsMul(E2I_NNZ, r_inh_i, plastic_delta, plastic_delta); '''
    elif sp_type == 'antihebb':
        p_types_names += [('const float', 'r_antihebb')]
        plasticity_local_floats += ['r_inh_anti[NI]']
        post_rates = '''
            cblas_scopy(NI, r_inh, 1, r_inh_anti, 1);
            //cblas_saxpy(NI,  1.f, &r_antihebb, 0, r_inh_anti, 1); //vsAdd(NI, r_inh, r_antihebb, r_inh_anti);
            for(int i = 0; i < NI; ++i)
                r_inh_anti[i] += r_antihebb;
            vmsInv(NI, r_inh_anti, r_inh_anti, VML_LA);
            // cblas_sscal(NI, r_antihebb_, r_inh_anti, 1);
            for(int i = 0; i < NI; ++i)
                r_inh_anti[i] *= r_antihebb;
            vsPackV(E2I_NNZ, r_inh_anti, e2i_data_i, r_inh_i); // Take equiv.
            vsMul(E2I_NNZ, r_inh_i, plastic_delta, plastic_delta);  '''
    else:
        post_rates = ''

    calculate_delta = f'''
            cblas_sscal(NE, eta, error, 1); // Scale the error by eta.
            vsPackV(E2I_NNZ, error, e2i_data_j, plastic_delta); // Take error into weight change array.
            {post_rates}
            plastic_nonlinearity_deriv(j_max, alpha_p, E2I_NNZ, plastic_data, plastic_logistic);
            vsMul(E2I_NNZ, plastic_logistic, plastic_delta, plastic_delta);
            '''
    update_rectify = '''
            vsAdd(E2I_NNZ, plastic_delta, plastic_data, plastic_data);
            plastic_nonlinearity(j_max, alpha_p, E2I_NNZ, plastic_data, e2i_data_pos);'''
    if plasticity_local_floats:
        plasticity_local_floats = 'static float {} __attribute__((aligned(ALIGNMENT))) = {{0}};'.format(
            ' __attribute__((aligned(ALIGNMENT))) = {0},\n                 '.join(plasticity_local_floats))
        plasticity_local_init = plasticity_local_floats + plasticity_local_init
    body = f'''
    {plasticity_local_init}
    {calculate_delta}
    {update_rectify}
    '''

    fn_name = 'update_weights'
    param_list = ', '.join(' '.join((t, n)) for t, n in p_types_names)
    fn_declaration = f'void {fn_name}({param_list})'
    define_update_function = f'''
{fn_declaration}
{{
    {body}
}}'''

    call_list = ', '.join(n for t, n in p_types_names)
    do_updates = f'{fn_name}({call_list});'
    return define_update_function, do_updates


def _non_local_plasticity(sp_type: str, sp_trans: float,
                          has_e2e: bool, has_i2i: bool,
                          do_ou_exc: bool, do_ou_inh: bool) -> Tuple[str, str]:
    p_types_names = [('float *', 'error'),
                     ('const float', 'eta'),
                     ('const float', 'j_max'),
                     ('float *', 'plastic_data'),
                     ('float *', 'plastic_delta')]
    if sp_trans == -4.:
        p_types_names += [('const float', 'rho0_inh_inv'),
                          ('const float*', 'r_inh'),
                          ('const float*', 'r_slow_inh'), ]
    else:
        p_types_names += [('const float', 'alpha_r'),
                          ('float *', 'h_inh'), ]
        if sp_trans == -3.:
            p_types_names += [('float *', 'h_slow_exc'),
                              ('float *', 'in_exc_state'),
                              ('float *', 'in_inh_state'),
                              ('float *', 'h_exc'),
                              ('const MKL_INT', 'n_e'),
                              ('const MKL_INT', 'n_i'),
                              ('const float *', 'i2e_data'),
                              ('const MKL_INT *', 'i2e_indptr'),
                              ('const MKL_INT *', 'i2e_indices'),
                              ('const float *', 'e2i_data_pos'),
                              ('const MKL_INT *', 'e2i_indptr'),
                              ('const MKL_INT *', 'e2i_indices'),
                              ('const float', 'bls_c'),
                              ('const float', 'bls_tau'),
                              ('const float', 'bls_alpha_lb'),
                              ('float *', 'eta_rec'),
                              ('float *', 'error_rec'),
                              ('float *', 'bls_t_rec')
                              ]
            if has_e2e:
                p_types_names += [('const float *', 'e2e_data'),
                                  ('const MKL_INT *', 'e2e_indptr'),
                                  ('const MKL_INT *', 'e2e_indices')]
            if has_i2i:
                p_types_names += [('const float *', 'i2i_data'),
                                  ('const MKL_INT *', 'i2i_indptr'),
                                  ('const MKL_INT *', 'i2i_indices')]
        elif sp_trans == -2.:
            p_types_names += [('const MKL_INT', 'n_e'),
                              ('float *', 'i2e_data_normed'),
                              ('const MKL_INT *', 'i2e_indptr'),
                              ('const MKL_INT *', 'i2e_indices'),
                              ]
            # if i2e_plastic:
            #     p_types_names += [('float *',     'i2e_data'),
            #                       ('const float', 'i2e_alpha'),
            #                       ('const float', 'i2e_thresh')]
        elif sp_trans == -1:
            p_types_names += [('const MKL_INT', 'n_i'),
                              ('float *', 'x2i_data_normed'),
                              ('const MKL_INT *', 'x2i_indptr'),
                              ('const MKL_INT *', 'x2i_indices')]
        else:  # Diffusive
            p_types_names += [('const DFTI_DESCRIPTOR_HANDLE', 'hand'),
                              ('const MKL_INT', 'e2i_stride')]

    if sp_type == 'inh_gain':
        p_types_names += [('float *', 'plastic_data_pos'),
                          ('const float *', 'inh_thresh'),
                          ('const float', 'alpha_p')]
        if sp_trans == -3.:
            p_types_names += [('const float *', 'inh_gain_pos')]
    elif sp_type == 'inh_thresh':
        p_types_names += [('const float *', 'inh_gain_pos'),
                          ('float *', 'inh_thresh')]
    elif sp_type in ('exc2inh', 'aff2inh'):
        p_types_names += [('float *', 'plastic_data_pos'),
                          ('const MKL_INT *', 'x2i_data_i'),
                          ('const MKL_INT *', 'x2i_data_j'),
                          ('const float*', 'r_pre'),
                          ('const float', 'alpha_p'), ]
        if sp_trans != -4.:
            p_types_names += [('const float *', 'inh_thresh'),
                              ('const float *', 'inh_gain_pos'), ]

    plasticity_local_floats = ['plastic_logistic[N_PLASTIC]'] if sp_type != 'inh_thresh' else []
    plasticity_local_init = '''
    bool error_increased = false;'''

    check_error = error_increased = error_decreased = ''
    calculate_i2i = '// No i2i to calculate'
    insert_i2i = '// No ii_eff to apply'
    calculate_e2e = '// No e2e to calculate'
    propagate_i2i = '// No ii_eff to propagate through'
    restore_ou_exc = restore_ou_inh = save_ou_exc = save_ou_inh = ''

    if sp_trans == -4.:  # BCM
        propagate_error = '''
        vsPowx(NI, r_slow_inh, 2, error);
        cblas_sscal(NI, rho0_inh_inv, error, 1);
        vsSub(NI, r_inh, error, error);
        vsMul(NI, r_inh, error, error);
        cblas_sscal(NI, eta, error, 1); // Scale the post term by eta.'''
    elif sp_trans == -3.:  # Full gradient
        plasticity_local_init += '''
    static float total_error_prev = INFINITY;
    float one_f = 1.f, zero_f = 0.f;
    float bls_alpha = eta, bls_t = 0.f;
    static MKL_INT ipiv[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                   eta_n = 0, info;
    const char *matdescra = "G__C_";'''
        plasticity_local_floats += ['dri_dhi[NI]', 'dre_dhe[NE]', 'w_eff[NE*NE]',
                                    'sqr_error[NE]', 'error_tmp[NE]', 'grad_buff[NE*NE]',
                                    'plastic_data_prev[N_PLASTIC]',
                                    'h_exc_prev[NE]', 'h_inh_prev[NI]',
                                    'h_slow_exc_prev[NE]', 'bls_p[N_PLASTIC]']
        if do_ou_exc:
            plasticity_local_floats += ['in_exc_state_prev[NE]']
            save_ou_exc = 'memcpy(in_exc_state_prev, in_exc_state, sizeof(in_exc_state_prev));'
            restore_ou_exc = 'memcpy(in_exc_state, in_exc_state_prev, sizeof(in_exc_state_prev));'
        if do_ou_inh:
            plasticity_local_floats += ['in_inh_state_prev[NI]']
            save_ou_inh = 'memcpy(in_inh_state_prev, in_inh_state, sizeof(in_inh_state_prev));'
            restore_ou_inh = 'memcpy(in_inh_state, in_inh_state_prev, sizeof(in_inh_state_prev));'
        if has_i2i:
            plasticity_local_floats += ['ii_eff[NI*NI]']
            plasticity_local_init += ''' 
    static MKL_INT ipiv_ii[NI] __attribute__((aligned(ALIGNMENT))) = {0};'''
            calculate_i2i = '''
        // First compute ii_eff
        for(int i = 0; i < NI; ++i)
            grad_buff[NI*i+i] = dri_dhi[i];
        // C := 1*A^T*B + 0*C
        // A: i2i          NIxNI
        // B: dri_dhi      NIxNI (inside grad_buff)
        // C: ii_eff       NIxNI
        mkl_scsrmm(do_trans, &n_i, // m: Number of rows of the matrix A. 
                   &n_i, // n: Number of columns of the matrix C. 
                   &n_i, // k: Number of columns in the matrix A.
                   &one_f, matdescra, 
                   i2i_data, i2i_indices, i2i_indptr, i2i_indptr+1, 
                   grad_buff, &n_i, // ldb:  Specifies the second dimension of B.
                   &zero_f, 
                   ii_eff, &n_i); // ldc: Specifies the second dimension of C.

        for(int i = 0; i < NI; ++i)
            ii_eff[NI*i+i] += 1.f;

        // Computes the LU factorization of a general m-by-n matrix. **Overwrites ii_eff**
        LAPACKE_sgetrf(LAPACK_ROW_MAJOR, NI, NI, ii_eff, NI, ipiv_ii);  '''
            insert_i2i = '''
        // Solves a system of linear equations with an LU-factored square coefficient matrix. 
        // A*X = B    **Overwrites w_eff_tmp**  NIxNI * NIxNE = NIxNE
        // We want ii_eff^-1 * w_eff_tmp = x -> Solve w_eff_tmp = ii_eff * x
        info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'T', // test change 'N', 
                       NI, // n: The order of A; the number of rows in B(n≥ 0).
                       NE, // nrhs: Number of right hand sides
                       ii_eff, NI, // lda: The leading dimension of a; lda>=max(1, n)
                       ipiv_ii, w_eff_tmp, NE // ldb: The leading dimension of b; ldb>=nrhs for row major layout.
                       );
        if( info > 0 ) {
                printf( "The diagonal element of the triangular factor of A,\\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\\n", info, info );
                printf( "the solution could not be computed.\\n" );
                exit( 1 );
        }'''
            propagate_i2i = '''
        info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', // test change 'T', 
                       NI, // n: The order of A; the number of rows in B(n≥ 0).
                       1, // nrhs: Number of right hand sides
                       ii_eff, NI, // lda: The leading dimension of a; lda>=max(1, n)
                       ipiv_ii, error, 1 // ldb: The leading dimension of b; ldb>=nrhs for row major layout.
                       );
        if( info > 0 ) {
                printf( "The diagonal element of the triangular factor of A,\\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\\n", info, info );
                printf( "the solution could not be computed.\\n" );
                exit( 1 );
        }'''
        if has_e2e:
            plasticity_local_floats += ['w_eff_tmp[NE*NE]']
            calculate_e2e = '''
        memset(grad_buff, 0, sizeof(grad_buff));
        for(int i = 0; i < NE; ++i)
            grad_buff[NE*i+i] = dre_dhe[i];
        // C := 1*A*B + 0*C
        // A: e2e          NExNE
        // B: dre_dhe      NExNE (inside grad_buff)
        // C: w_eff_tmp    NExNE
        mkl_scsrmm(no_trans, &n_e, // m: Number of rows of the matrix A. 
                   &n_e, // n: Number of columns of the matrix C. 
                   &n_e, // k: Number of columns in the matrix A.
                   &one_f, matdescra, 
                   e2e_data, e2e_indices, e2e_indptr, e2e_indptr+1, 
                   grad_buff, &n_e, // ldb:  Specifies the second dimension of B.
                   &zero_f, 
                   w_eff_tmp, &n_e); // ldc: Specifies the second dimension of C.
        vsSub(NE*NE, w_eff, w_eff_tmp, w_eff);'''
        else:
            plasticity_local_floats += ['w_eff_tmp[NI*NE]']

        check_error = '''
    vsPowx(NE, error, 2.f, sqr_error);
    float total_error = cblas_sasum(NE, sqr_error, 1) / 2.f;        

    // Perform Backtracking Line Search
    float bls_delta = total_error_prev - total_error, // Want this positive
          bls_thresh = bls_alpha * bls_t; // If gradient was large, expect larger decrease in error
    error_increased = bls_delta <= bls_thresh; // If change is left of our expected decrease, we say error increased.
    eta_rec[eta_n] = bls_alpha;
    bls_t_rec[eta_n] = bls_t;
    error_rec[eta_n++] = total_error;
    if(bls_alpha < bls_alpha_lb)
    {
        // If alpha is too small, we reset it and recompute the gradient.
        printf("Resetting bls_alpha, recomputing gradient");
        error_increased = false;
    }'''
        error_increased = '''
        // If the error increased, undo the change and decrease step size.
        bls_alpha *= bls_tau;

        // Compute a new delta and apply
        cblas_scopy(N_PLASTIC, bls_p, 1, plastic_delta, 1);
        cblas_sscal(N_PLASTIC, bls_alpha, plastic_delta, 1);
        memcpy(plastic_data, plastic_data_prev, sizeof(plastic_data_prev)); 
        {update_rectify}

        // Restore remaining state
        memcpy(h_slow_exc, h_slow_exc_prev, sizeof(h_slow_exc_prev));
        memcpy(h_exc, h_exc_prev, sizeof(h_exc_prev));
        memcpy(h_inh, h_inh_prev, sizeof(h_inh_prev));
        {restore_ou_exc}
        {restore_ou_inh}  '''
        error_decreased = '''
        // If the error decreased, restore step size and save state before recomputing gradient
        bls_alpha = eta;
        total_error_prev = total_error;

        // Store current state before update
        memcpy(plastic_data_prev, plastic_data, sizeof(plastic_data_prev));
        memcpy(h_slow_exc_prev, h_slow_exc, sizeof(h_slow_exc_prev)); 
        memcpy(h_exc_prev, h_exc, sizeof(h_exc_prev));
        memcpy(h_inh_prev, h_inh, sizeof(h_inh_prev));
        {save_ou_exc}
        {save_ou_inh}  '''
        propagate_error = '''
        // compute w_eff = (1+w_i2e dri_dhi w_e2i dre_dhe)
        // or w_eff = (1+w_i2e ii_eff dri_dhi w_e2i dre_dhe + w_e2e dre_dhe)
        memset(grad_buff, 0, sizeof(grad_buff));

        rate_nonlinearity_deriv(f_max, alpha_r, NI, h_inh, dri_dhi, inh_thresh, inh_gain_pos, OTHER);  // Fix this line
        pe2("dri_dhi", dri_dhi, 1, NI);
        {calculate_i2i}

        rate_nonlinearity_deriv(f_max, alpha_r, NE, h_exc, dre_dhe, NULL, NULL, OTHER);
        pe2("dre_dhe", dre_dhe, 1, NE);

        //if(0)
        //    vsMul(NE, dre_dhe, error, error);

        // fill dense matrix grad_buff NExNE
        for(int i = 0; i < NE; ++i)
            grad_buff[NE*i+i] = dre_dhe[i];

        // C := 1*A*B + 0*C
        // A: e2i          NIxNE
        // B: dre_dhe      NExNE (inside grad_buff)
        // C: w_eff_tmp    NIxNE
        mkl_scsrmm(no_trans, &n_i, // m: Number of rows of the matrix A. 
                   &n_e, // n: Number of columns of the matrix C. 
                   &n_e, // k: Number of columns in the matrix A.
                   &one_f, matdescra, 
                   e2i_data_pos, e2i_indices, e2i_indptr, e2i_indptr+1, 
                   grad_buff, &n_e, // ldb:  Specifies the second dimension of B.
                   &zero_f, 
                   w_eff_tmp, &n_e); // ldc: Specifies the second dimension of C.
        pe2("wh", w_eff_tmp, NI, n_e);

        // diag(dri_dhi)*e2i*diag(dre_dhe)
        for(int i = 0; i < NI; ++i)
            cblas_sscal(NE, dri_dhi[i], &w_eff_tmp[i*NE], 1);
        pe2("gwh", w_eff_tmp, NI, NE);
        {insert_i2i}

        // C:= 1*A*B + 0*C
        // A: i2e            NExNI
        // B: w_eff_tmp     NIxNE
        // C: w_eff          NExNE
        mkl_scsrmm(no_trans , &n_e , &n_e , &n_i, 
                   &one_f, matdescra, 
                   i2e_data, i2e_indices, i2e_indptr, i2e_indptr+1, 
                   w_eff_tmp, &n_e, &zero_f, 
                   w_eff, &n_e);

        pe2("wgw", w_eff, NE, NE);

        for(int i = 0; i < NE; ++i)
            w_eff[NE*i+i] += 1.f;

        pe2("w_inv", w_eff, NE, NE);
        {calculate_e2e}

        // Now that we have w_eff, we use it to compute error*(1+ww)^-1

        // Computes the LU factorization of a general m-by-n matrix. **Overwrites w_eff**
        LAPACKE_sgetrf(LAPACK_ROW_MAJOR, NE, NE, w_eff, NE, ipiv);

        // Solves a system of linear equations with an LU-factored square coefficient matrix. 
        // A^T*X = B    **Overwrites error**
        // We want r_eff = error^T*w_eff^-1 = w_eff^-T*error
        // w_eff^T*r_eff = error (A^T*X = B) 
        info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'T', 
                       NE, // n: The order of A; the number of rows in B(n≥ 0).
                       1, // nrhs: Number of right hand sides
                       w_eff, NE, // lda: The leading dimension of a; lda>=max(1, n)
                       ipiv, error, 1 // ldb: The leading dimension of b; ldb>=nrhs for row major layout.
                       );
        if( info > 0 ) {{
                printf( "The diagonal element of the triangular factor of A,\\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\\n", info, info );
                printf( "the solution could not be computed.\\n" );
                exit( 1 );
        }}

        pe2("error_rec", error, 1, NE);

        // error now contains the error evaluated through the recurrent connections.
        // We now back-trace that error through the inhibitory connectivity. [error*(1+ww diag)^-1] * wei
        mkl_cspblas_scsrgemv(do_trans, &n_e, i2e_data, i2e_indptr, i2e_indices, error, error_tmp);
        cblas_scopy(NI, error_tmp, 1, error, 1);
        {propagate_i2i}

        pe2("backprop_err", error, 1, NI);  '''
    elif sp_trans == -2.:  # Retrograde
        plasticity_local_floats += ['error_temp[NE]']
        # No need to re-norm here, done on update of i2e
        # if i2e_plastic:
        #     # Need to re-normalize the i2e data every round due to presence of iSP.
        #     do_norm = '''
        # cblas_scopy(I2E_NNZ, i2e_data, 1, i2e_data_normed, 1);
        # cblas_saxpy(I2E_NNZ, -1.f, &i2e_thresh, 0, i2e_data_normed, 1);
        # derivative_nl(i2e_alpha, I2E_NNZ, i2e_data_normed, i2e_data_normed);'''
        # else:
        #     do_norm = ''

        propagate_error = '''
        // Retrograde transmission of error.
        mkl_cspblas_scsrgemv(do_trans, &n_e, i2e_data_normed, i2e_indptr, i2e_indices, error, error_temp);
        cblas_scopy(NI, error_temp, 1, error, 1);
        cblas_sscal(NI, eta, error, 1); // Scale the error by eta.'''
    elif sp_trans == -1:  # Anterograde
        plasticity_local_floats += ['error_temp[NE]']  # 'e2i_data_normed[E2I_NNZ]']
        propagate_error = '''
        // Anterograde transmission of error.
        mkl_cspblas_scsrgemv(no_trans, &n_i, x2i_data_normed, x2i_indptr, x2i_indices, error, error_temp);
        cblas_scopy(NI, error_temp, 1, error, 1);
        cblas_sscal(NI, eta, error, 1); // Scale the error by eta.'''
    else:  # Diffusive
        propagate_error = '''
        spatial_filter(hand, error, error);
        cblas_scopy(NI, error, e2i_stride, error, 1); // Down-sample NE:NI
        cblas_sscal(NI, eta, error, 1); // Scale the error by eta.'''

    if sp_type == 'inh_gain':
        plasticity_local_init += '''
    plastic_nonlinearity(j_max, alpha_p, NI, plastic_data, plastic_data_pos);'''
        plasticity_local_floats += ['dri_dgi[NI]']
        calculate_delta = '''
        cblas_scopy(NI, error, 1, plastic_delta, 1); // Take error into weight change array.

        // Compute dr_inh/dg_inh (thresh is 1)
        assert(0); // Fix the following line for the new non-linearity
        rate_nonlinearity(alpha_r, NI, h_inh, dri_dgi, inh_thresh, NULL);
        vsMul(NI, dri_dgi, plastic_delta, plastic_delta);

        // derivative of gain wrt to v, enforces non-negativity
        plastic_nonlinearity_deriv(j_max, alpha_p, NI, plastic_data, plastic_logistic);
        vsMul(NI, plastic_logistic, plastic_delta, plastic_delta);   '''
        update_rectify = '''
        vsAdd(NI, plastic_delta, plastic_data, plastic_data);
        plastic_nonlinearity(j_max, alpha_p, NI, plastic_data, plastic_data_pos);'''
    elif sp_type == 'inh_thresh':
        plasticity_local_floats += ['dri_dti[NI]']
        calculate_delta = '''
        cblas_scopy(NI, error, 1, plastic_delta, 1); // Take error into weight change array.

        // Compute dr_inh/d_thresh (gain is 1)            
        assert(0); // Fix the following line
        rate_nonlinearity_deriv(f_max, alpha_r, NI, h_inh, dri_dti, inh_thresh, inh_gain_pos, THRESH);
        // Multiply delta by dr_inh/d_thresh            
        vsMul(NI, dri_dti, plastic_delta, plastic_delta);

        // derivative of thresh wrt to v is 1, enforces no constraint on negativity'''
        update_rectify = '''
        vsAdd(NI, plastic_data, plastic_delta, plastic_data);
        clip(NI, plastic_data, -j_max, j_max);'''
    elif sp_type in ('exc2inh', 'aff2inh'):
        if sp_trans == -4.:
            dri_dhi = ''
        else:
            if sp_trans != -3.:
                plasticity_local_floats += ['dri_dhi[NI]']
                p_types_names += [('const float', 'f_max'), ]
                dri_dhi = 'rate_nonlinearity_deriv(f_max, alpha_r, NI, h_inh, dri_dhi, inh_thresh, inh_gain_pos, OTHER);'
            else:
                dri_dhi = '// dri_dhi already calculated during error propagation'
            dri_dhi += '''
        // Prefer synapses onto active interneurons.
        vsMul(NI, dri_dhi, error, error);'''

        calculate_delta = f'''
        {dri_dhi}
        // Prefer synapses driven by cells with high rates
        vsPackV(N_PLASTIC, r_pre, x2i_data_j, plastic_logistic); 
        vsPackV(N_PLASTIC, error, x2i_data_i, plastic_delta); // Take error into weight change array.
        vsMul(N_PLASTIC, plastic_logistic, plastic_delta, plastic_delta); // Multiply error by pre-rates.
        // derivative of e2i wrt v, enforces non-negativity.
        plastic_nonlinearity_deriv(j_max, alpha_p, N_PLASTIC, plastic_data, plastic_logistic);

        vsMul(N_PLASTIC, plastic_logistic, plastic_delta, plastic_delta);'''
        update_rectify = '''
        vsAdd(N_PLASTIC, plastic_delta, plastic_data, plastic_data);
        plastic_nonlinearity(j_max, alpha_p, N_PLASTIC, plastic_data, plastic_data_pos);'''
        if sp_trans == -1.:
            p_types_names += [('const float', 'x2i_thresh'),
                              ('float *', 'num_above_thresh')]
            plasticity_local_init += '''
        x2i_data_normed = __builtin_assume_aligned(x2i_data_normed, ALIGNMENT);
        plastic_data_pos = __builtin_assume_aligned(plastic_data_pos, ALIGNMENT);'''
            n_x = 'NE' if sp_type == 'exc2inh' else 'N_A2I'
            update_rectify += f'''

        for(int i = 0; i < N_PLASTIC; ++i)
            x2i_data_normed[i] = plastic_data_pos[i] > x2i_thresh ? 1.f : 0.f;
        sum_axis(ROW_WISE, x2i_data_normed, NI, {n_x}, x2i_indices, x2i_indptr, num_above_thresh);
        div_by_sums(ROW_WISE, x2i_data_normed, N_PLASTIC, NI, {n_x}, 
                    x2i_indices, x2i_indptr, num_above_thresh, x2i_data_normed);'''
    else:
        raise Exception(f'Plasticity not implemented ({sp_type})')

    if sp_trans == -3.:
        calculate_delta += '''
                // Store the normalized gradient in bls_p, and compute the update threshold.
                cblas_scopy(N_PLASTIC, plastic_delta, 1, bls_p, 1);
                cblas_sscal(N_PLASTIC, 1.f / cblas_snrm2(N_PLASTIC, bls_p, 1), bls_p, 1); // Normalize gradient
                bls_t = bls_c * cblas_sdot(N_PLASTIC, bls_p, 1, plastic_delta, 1); // Compute threshold t for Armijo-Goldstein
                cblas_scopy(N_PLASTIC, bls_p, 1, plastic_delta, 1);
                cblas_sscal(N_PLASTIC, bls_alpha, plastic_delta, 1);
                '''
        error_increased = error_increased.format(update_rectify=update_rectify,
                                                 restore_ou_exc=restore_ou_exc, restore_ou_inh=restore_ou_inh)
        error_decreased = error_decreased.format(save_ou_exc=save_ou_exc, save_ou_inh=save_ou_inh)

    propagate_error = propagate_error.format(calculate_i2i=calculate_i2i, calculate_e2e=calculate_e2e,
                                             insert_i2i=insert_i2i, propagate_i2i=propagate_i2i)

    if plasticity_local_floats:
        plasticity_local_floats = 'static float {} __attribute__((aligned(ALIGNMENT))) = {{0}};'.format(
            ' __attribute__((aligned(ALIGNMENT))) = {0},\n                 '.join(plasticity_local_floats))
        plasticity_local_init = plasticity_local_floats + plasticity_local_init
    body = f'''
    {plasticity_local_init}
    {check_error}
    if(!error_increased)
    {{
        {error_decreased}
        {propagate_error}
        {calculate_delta}
        {update_rectify}
    }}
    else
    {{
        {error_increased}
    }}'''

    fn_name = 'update_plastic'
    param_list = ', '.join(' '.join((t, n)) for t, n in p_types_names)
    fn_declaration = f'void {fn_name}({param_list})'
    define_update_function = f'''
{fn_declaration}
{{
    {body}
}}'''

    call_list = ', '.join(n for t, n in p_types_names)
    call_update = f'{fn_name}({call_list});'

    return define_update_function, call_update


def _weight_update(sp_type: str, i2e_plastic: bool, calibrate_syn: str, sp_trans: float,
                   has_e2e: bool, has_i2i: bool, do_ou_exc: bool, do_ou_inh: bool,
                   weight_update_on_steps: bool, i2e_binary: bool,
                   do_print_arrays: bool, do_online_report: bool) -> weight_updater:
    i2e_threshold = 1
    i2e_normalize = 1
    plasticity_floats = []

    err_term = 'h_slow_exc' if sp_trans == -3. else 'r_slow_exc'
    if sp_type != 'none':
        plasticity_floats += ['plastic_delta[N_PLASTIC]']

    plasticity_init = ''
    if sp_type != 'inh_thresh':
        plasticity_init += '''
    const float *inh_thresh = NULL;'''
    if sp_type == 'inh_gain':
        plasticity_floats += ['inh_gain_pos[NI]']
    else:
        plasticity_init += '''
    const float *inh_gain_pos = NULL;'''

    if sp_type == 'none':
        plasticity_init += '''
    printf("NO plasticity for interneuron activity\\n");'''
    elif sp_trans >= 0. and sp_type in ('exc2inh', 'inh_gain', 'inh_thresh', 'aff2inh'):
        plasticity_init += f'''
    DFTI_DESCRIPTOR_HANDLE hand = init_dfti(k);
    const MKL_INT e2i_stride = NE/NI; // Always down-sample
    printf("Using {sp_type} DIFFUSIVE plasticity with max %f, down-sample stride %d\\n", (double)j_max, e2i_stride);
    assert(e2i_stride > 0);'''
    elif sp_trans == -1:
        plasticity_init += f'''
    printf("Using {sp_type} ANTEROGRADE plasticity with max %f\\n", (double)j_max);'''
    elif sp_trans == -2:
        plasticity_init += f'''
    normalize_sparse(COL_WISE, i2e_data, I2E_NNZ, NE, NI, i2e_indices, i2e_indptr, i2e_data_normed);
    printf("Using {sp_type} RETROAXONAL plasticity with max %f\\n", (double)j_max);'''
    elif sp_trans == -3:
        plasticity_init += f'''
    printf("Using full {sp_type} GRADIENT rule with max %f \\n", (double)j_max);'''
    elif sp_trans == -4.:
        plasticity_init += '''
    const float rho0_inh_inv = 1.f/rho0_inh;
    printf("Using BCM plasticity with rho0 %f eta %g, max %f\\n", (double)rho0_inh, (double)eta, (double)j_max);'''
    if i2e_plastic:
        plasticity_init += '''
    printf("INHIBITORY SYNAPTIC plasticity is enabled, with max %f \\n", (double)j_i2e_max); '''

    inh2exc = ''
    update_functions = ''
    call_updates = ''
    if sp_type in ('nonhebb', 'hebbian', 'antihebb'):
        define_update_function, call_update = _local_plasticity(sp_type)
        update_functions += define_update_function
        call_updates += call_update
    # elif sp_type == 'bcm':
    #     '''
    #     Version of BCM without explicit target rate, to have one use exc2inh -4
    #
    #     tau tv' = v - tv
    #     tv(t) = c1 exp(-t/tau) + exp(-t/tau)/tau int_1^t exp(x/tau) v(x) dx
    #
    #     tau tv' = v^2 - tv
    #     tv(t) = c1 exp(-t/tau) + exp(-t/tau)/tau int_1^t exp(x/tau) v(x)^2 dx
    #     '''
    #     plasticity_floats += ['r_post_i[E2I_NNZ]',
    #                           'r_post_tmp[NI]']
    #     plasticity_init += '''
    #     printf("Using BCM plasticity with eta %g, max %f\\n", (double)eta, (double)j_max);'''
    #     calculate_delta = '''
    #             vsSub(NI, r_inh, r_slow_inh2, r_post_tmp);
    #             vsMul(NI, r_inh, r_post_tmp, r_post_tmp);
    #             if(1)
    #                 vsDiv(NI, r_post_tmp, r_slow_inh2, r_post_tmp);
    #             else
    #             {
    #                 derivative_nl(alpha_r, NI, r_inh, r_post_i);
    #                 vsMul(NI, r_post_i, r_post_tmp, r_post_tmp);
    #             }
    #             cblas_sscal(NI, eta, r_post_tmp, 1); // Scale the post term by eta.
    #
    #             vsPackV(E2I_NNZ, r_post_tmp, e2i_data_i, r_post_i); // Take post term into temp array.
    #             vsPackV(E2I_NNZ, r_exc, e2i_data_j, plastic_delta); // Take pre term into weight change array.
    #             vsMul(E2I_NNZ, r_post_i, plastic_delta, plastic_delta); // Multiply post with pre terms.  '''
    #     update_rectify = '''
    #             vsAdd(E2I_NNZ, plastic_delta, e2i_data, e2i_data);
    #             clip(E2I_NNZ, e2i_data, 0.f, j_max);
    #             cblas_scopy(E2I_NNZ, e2i_data, 1, e2i_data_pos, 1);'''
    # elif sp_type == 'none' and not i2e_plastic:
    #     update_rectify = '''
    #     vsAdd(N_PLASTIC, plastic_delta, plastic_data, plastic_data);
    #     clip_upper(N_PLASTIC, plastic_data, j_max);
    #     plastic_nonlinearity(alpha_p, N_PLASTIC, plastic_data, plastic_data_pos);'''
    elif sp_type != 'none':
        define_update_function, call_update = _non_local_plasticity(sp_type, sp_trans,
                                                                    has_e2e, has_i2i,
                                                                    do_ou_exc, do_ou_inh)
        update_functions += define_update_function
        call_updates += call_update

    if i2e_plastic:
        plasticity_floats += ['i2e_delta[I2E_NNZ]',
                              'i2e_pre_rates[I2E_NNZ]']
        inh2exc = '''
            vsPackV(I2E_NNZ, error, i2e_data_i, i2e_delta); // Take post-synaptic error into weight change array.
            vsPackV(I2E_NNZ, r_inh, i2e_data_j, i2e_pre_rates); // Take pre-synaptic inh rates
            vsMul(I2E_NNZ, i2e_pre_rates, i2e_delta, i2e_delta);
            cblas_sscal(I2E_NNZ, eta_i2e, i2e_delta, 1); // Scale the weight change by eta.
            vsAdd(I2E_NNZ, i2e_delta, i2e_data, i2e_data);
            for(MKL_INT i = 0; i < I2E_NNZ; ++i) i2e_data[i] = i2e_data[i] < 0.f ? 0.f : i2e_data[i];
            for(MKL_INT i = 0; i < I2E_NNZ; ++i) i2e_data[i] = i2e_data[i] > j_i2e_max ? j_i2e_max : i2e_data[i];'''
        if sp_trans == -2. and sp_type != 'none':
            inh2exc += '''
            cblas_scopy(I2E_NNZ, i2e_data, 1, i2e_data_normed, 1);'''
            if i2e_threshold:
                data = '1.f' if i2e_binary else 'i2e_data_normed[i]'
                inh2exc += f'''
            for(MKL_INT i = 0; i < I2E_NNZ; ++i) 
                i2e_data_normed[i] = i2e_data_normed[i] < i2e_thresh ? 0.f : {data};'''

            if i2e_normalize:
                inh2exc += '''
            normalize_sparse(COL_WISE, i2e_data, I2E_NNZ, NE, NI, i2e_indices, i2e_indptr, i2e_data_normed);'''

    if sp_type in ('exc2inh', 'aff2inh'):
        x_ae = 'e' if sp_type == 'exc2inh' else 'a'
        plasticity_init += f'''
    float *plastic_data_pos = __builtin_assume_aligned({x_ae}2i_data_pos, ALIGNMENT);
    const MKL_INT *x2i_data_i = {x_ae}2i_data_i, *x2i_data_j = {x_ae}2i_data_j;'''
        if sp_type == 'exc2inh':
            plasticity_init += '''
    const float *r_pre = r_exc;'''
        if sp_trans == -1.:
            n_x = 'NE' if sp_type == 'exc2inh' else 'N_A2I'
            plasticity_floats += ['x2i_data_normed[N_PLASTIC]',
                                  'num_above_thresh[NI]',
                                  'num_presynaptic[NI]']
            plasticity_init += f'''
    const MKL_INT *x2i_indices = {x_ae}2i_indices, *x2i_indptr = {x_ae}2i_indptr;        
    for(int i = 0; i < N_PLASTIC; ++i)
        x2i_data_normed[i] = 1.f;
    sum_axis(ROW_WISE, x2i_data_normed, NI, {n_x}, x2i_indices, x2i_indptr, num_presynaptic);

    for(int i = 0; i < N_PLASTIC; ++i)
        x2i_data_normed[i] = plastic_data_pos[i] > x2i_thresh ? 1.f : 0.f;
    sum_axis(ROW_WISE, x2i_data_normed, NI, {n_x}, x2i_indices, x2i_indptr, num_above_thresh);
    div_by_sums(ROW_WISE, x2i_data_normed, N_PLASTIC, NI, {n_x}, x2i_indices, x2i_indptr, 
                num_above_thresh, x2i_data_normed);'''
    if sp_type == 'inh_gain':
        plasticity_init += '''
    float * plastic_data_pos = inh_gain_pos;'''

    if sp_type in ('hebbian', 'nonhebb', 'antihebb'):
        plasticity_init += '''
    float *plastic_data_pos = e2i_data_pos;'''

    if sp_type == 'none':
        if i2e_plastic:
            plasticity_init += '''
    float *plastic_data_pos = i2e_data;'''
        else:
            plasticity_init += '''
    float *plastic_data_pos = e2i_data;'''

    calibrate_weights = 'float r_mu_error = cblas_sasum(NE, r_exc, 1) / NE - rho_calibrate;'
    if 'e2i' in calibrate_syn:
        calibrate_weights += '''
            cblas_saxpy(E2I_NNZ, calibration_eta*NI/NE, &r_mu_error, 0, e2i_data, 1);
            plastic_nonlinearity(j_max, alpha_p, E2I_NNZ, e2i_data, e2i_data_pos);'''
    if 'i2e' in calibrate_syn:
        calibrate_weights += '''
            cblas_saxpy(I2E_NNZ, calibration_eta, &r_mu_error, 0, i2e_data, 1);
            clip(I2E_NNZ, i2e_data, 0.f, j_i2e_max);'''
    if 'a2i' in calibrate_syn:
        calibrate_weights += '''
            cblas_saxpy(A2I_NNZ, calibration_eta*NI/NE, &r_mu_error, 0, a2i_data, 1);
            plastic_nonlinearity(j_max, alpha_p, A2I_NNZ, a2i_data, a2i_data_pos);'''

    print_arrays = '''
            printf("eta = np.loadtxt(StringIO('%.6a'), dtype=np.float32)\\n", (double)eta);
            pe2("r_exc", r_exc, 1, NE);
            pe2("r_inh", r_inh, 1, NI);
            //pia("e2i_data_i", e2i_data_i, E2I_NNZ);
            //pia("e2i_data_j", e2i_data_j, E2I_NNZ);
            pe2("dw_", plastic_delta, 1, N_PLASTIC);
            psa("e2i", e2i_data, E2I_NNZ, NI, NE, e2i_indices, e2i_indptr);
            psa("i2e", i2e_data, I2E_NNZ, NE, NI, i2e_indices, i2e_indptr);
            pe2("e2i_data_pos", e2i_data_pos, 1, NI);'''

    if plasticity_floats:
        plasticity_floats = 'static float {} __attribute__((aligned(ALIGNMENT))) = {{0}};'.format(
            ' __attribute__((aligned(ALIGNMENT))) = {0},\n                 '.join(plasticity_floats))
        plasticity_init = plasticity_floats + plasticity_init

    print_arrays = print_arrays if do_print_arrays else ''
    if weight_update_on_steps:
        update_condition = 'update_weights_n[wu_idx] == n && wu_idx++'
        plasticity_init += '''
    MKL_INT wu_idx = 0;'''
    else:
        update_condition = 'n % update_weights_n == 0'
    update_weights = f'''
        if({update_condition} && plasticity_on <= n && n < plasticity_off)
        {{
            cblas_scopy(NE, {err_term}, 1, error, 1); // Compute the per-excitatory neuron error.
            cblas_saxpy(NE,  1.f, &minus_rho0, 0, error, 1);
            {inh2exc}
            {call_updates}
            {print_arrays}
        }}
        else if(n < n_calibrate && n % 10 == 0)
        {{
            {calibrate_weights}
        }}
    '''

    return weight_updater(plasticity_init, update_weights, update_functions)


def _inner_fn(sp_type: str, sp_trans: float, plastic_var: str, weight_update_on_steps: bool, i2e_binary: bool,
              has_k: bool, i2e_plastic: bool, calibrate_syn: str,
              do_ou_exc: bool, do_ou_inh: bool, do_aff_inh: bool, has_e2e: bool, has_i2i: bool,
              do_print_arrays: bool, rec_all_plastic: bool, do_mmap: bool) -> function_definitions:
    # global all_parameters
    # cblas_sasum(NI, inh_tmp, 1) / NI
    # //sparse_matrix_t  e2i, i2e;
    # //sparse_status_t ss = mkl_sparse_s_create_csr(&e2i, sparse_index_base_t indexing, NI, NE,
    # // sparse_status_t mkl_sparse_destroy( sparse_matrix_t A);
    # //                                            MKL_INT *pntrb,  MKL_INT *pntre,  MKL_INT *indx, float *val );
    # // pointerB[i] = rowIndex[i]
    # // pointerE[i] = rowIndex[i+1]
    # // This enables calling a routine that has values, columns, pointerB and pointerE as input parameters for a sparse
    # // matrix stored in the format accepted for the direct sparse solvers. For example, a routine with the interface:
    # // void name_routine(.... ,  double *values, MKL_INT *columns, MKL_INT *pointerB, MKL_INT *pointerE, ...)
    # // can be called with parameters values, columns, rowIndex as follows:
    # // name_routine(.... ,  values, columns, rowIndex, rowIndex+1, ...).

    mkl_threads = 1
    do_online_report = False

    do_aff_exc = 0

    if do_ou_exc:
        init_exc_input = '''
    float in_ou_exc_sigma_tau = sqrt(2 * dt * (float)pow(in_ou_exc_sigma, 2) / in_ou_exc_tau),
          norm_exc[NE] __attribute__((aligned(ALIGNMENT))),
          in_ou_exc_dt_tau = dt / in_ou_exc_tau;
    printf("Using OU input for excitatory population mu: %f sigma: %f \\n", (double)in_exc_mean[0], (double)in_ou_exc_sigma);'''
        update_exc_input = '''
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, ou_stream, NE, norm_exc, 0, in_ou_exc_sigma_tau);
        cblas_saxpy(NE, -in_ou_exc_dt_tau, in_exc_state, 1, norm_exc, 1); 
        vsAdd(NE, norm_exc, in_exc_state, in_exc_state); '''
        # // vsSub(NE, &in_exc_mean[NE*curr_pattern_index], in_exc_state, in_exc_tmp);
        # // cblas_sscal(NE, in_ou_exc_dt_tau, in_exc_tmp, 1);
        # // vsAdd(NE, in_exc_tmp, in_exc_state, in_exc_state);
    else:
        update_exc_input = ''
        init_exc_input = ''
    if do_ou_inh:
        init_inh_input = '''
    float in_ou_inh_sigma_tau = sqrt(2 * dt * (float)pow(in_ou_inh_sigma, 2) / in_ou_inh_tau),
          norm_inh[NI] __attribute__((aligned(ALIGNMENT))),
          in_ou_inh_dt_tau = dt / in_ou_inh_tau;
    printf("Using OU input for inhibitory population mu: %f sigma: %f \\n", (double)in_inh_mean[0], (double)in_ou_inh_sigma);'''
        update_inh_input = '''
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, ou_stream, NI, norm_inh, 0, in_ou_inh_sigma_tau);
        cblas_saxpy(NI, -in_ou_inh_dt_tau, in_inh_state, 1, norm_inh, 1); 
        vsAdd(NI, norm_inh, in_inh_state, in_inh_state);'''
        # // vsSub(NI, &in_inh_mean[NE*curr_pattern_index], in_inh_state, in_inh_tmp);
        # // cblas_sscal(NI, in_ou_inh_dt_tau, in_inh_tmp, 1);
    else:
        update_inh_input = ''
        init_inh_input = ''

    update_inputs = f'''
    {update_exc_input}
    {update_inh_input} '''
    if do_ou_exc or do_ou_inh:
        ou_init = f'''
    VSLStreamStatePtr ou_stream;
    vslNewStream( &ou_stream, VSL_BRNG_SFMT19937, seed);
    printf("Init SFMT19937 stream using seed value %d \\n", seed);
    {init_exc_input}
    {init_inh_input} '''
        ou_cleanup = 'vslDeleteStream( &ou_stream );'
    else:
        update_inputs = ''
        ou_init = 'printf("No OU inputs requested.\\n");'
        ou_cleanup = ''

    a2e_init = ''
    update_exc = ''
    if do_aff_exc:
        a2e_init += '''
    static float a2e_data_pos[A2E_NNZ] __attribute__((aligned(ALIGNMENT))) = {0};
    plastic_nonlinearity(j_max, alpha_p, A2E_NNZ, a2e_data, a2e_data_pos);'''
        update_exc += '''
        mkl_cspblas_scsrgemv(no_trans, &n_e, a2e_data_pos, a2e_indptr, a2e_indices, &in_exc_patterns[NE*curr_pattern_index], in_exc_current); 
        vsAdd(NE, in_exc_state, in_exc_current, in_exc_current);
        vsAdd(NE, in_exc_mean, in_exc_current, in_exc_current);'''
    else:
        update_exc += '''
        vsAdd(NE, &in_exc_patterns[NE*curr_pattern_index], in_exc_state, in_exc_current);
        vsAdd(NE, in_exc_mean, in_exc_current, in_exc_current);'''
    update_exc += '''
        mkl_cspblas_scsrgemv(no_trans, &n_e, i2e_data, i2e_indptr, i2e_indices, r_inh, h_2e);
        vsSub(NE, in_exc_current, h_2e, in_exc_current);'''
    if has_e2e:
        update_exc += '''
        mkl_cspblas_scsrgemv(no_trans, &n_e, e2e_data, e2e_indptr, e2e_indices, r_exc, h_2e);
        vsAdd(NE, h_2e, in_exc_current, in_exc_current); '''
    update_exc += '''
        vsSub(NE, in_exc_current, h_exc, in_exc_current);        // Compute change to input, dh = h_exc(t+1) - h_exc(t)
        cblas_saxpy(NE, dt_tau_exc, in_exc_current, 1, h_exc, 1); // New input h_exc += dt*dh/tau
        rate_nonlinearity(f_max, alpha_r, NE, h_exc, r_exc, NULL, NULL);'''

    a2i_init = ''
    update_inh = ''
    if do_aff_inh:
        a2i_init += '''
    static float a2i_data_pos[A2I_NNZ] __attribute__((aligned(ALIGNMENT))) = {0};
    plastic_nonlinearity(j_max, alpha_p, A2I_NNZ, a2i_data, a2i_data_pos);'''
        if sp_type == 'aff2inh':
            update_inh += '''
        const float *r_pre = &in_inh_patterns[N_A2I*curr_pattern_index];
        mkl_cspblas_scsrgemv(no_trans, &n_i, a2i_data_pos, a2i_indptr, a2i_indices, r_pre, in_inh_current);'''
        else:
            update_inh += '''
        mkl_cspblas_scsrgemv(no_trans, &n_i, a2i_data_pos, a2i_indptr, a2i_indices, &in_inh_patterns[N_A2I*curr_pattern_index], in_inh_current);'''
        update_inh += '''
        vsAdd(NI, in_inh_state, in_inh_current, in_inh_current);
        vsAdd(NI, in_inh_mean, in_inh_current, in_inh_current);'''
    else:
        update_inh += '''
        cblas_scopy(NI, in_inh_state, 1, in_inh_current, 1);
        vsAdd(NI, in_inh_mean, in_inh_current, in_inh_current);'''

    update_inh += '''
        mkl_cspblas_scsrgemv(no_trans, &n_i, e2i_data_pos, e2i_indptr, e2i_indices, r_exc, h_2i);
        vsAdd(NI, h_2i, in_inh_current, in_inh_current);'''
    if has_i2i:
        update_inh += '''
        mkl_cspblas_scsrgemv(no_trans, &n_i, i2i_data, i2i_indptr, i2i_indices, r_inh, h_2i);
        vsSub(NI, in_inh_current, h_2i, in_inh_current); '''
    update_inh += '''
        vsSub(NI, in_inh_current, h_inh, in_inh_current);
        cblas_saxpy(NI, dt_tau_inh, in_inh_current, 1, h_inh, 1); // New input h_inh += dt*dh/tau
        rate_nonlinearity(f_max, alpha_r, NI, h_inh, r_inh, inh_thresh, inh_gain_pos);'''

    if sp_type == 'bcm':
        make_e2i_pos = 'cblas_scopy(E2I_NNZ, e2i_data, 1, e2i_data_pos, 1);'
    else:
        make_e2i_pos = 'plastic_nonlinearity(j_max, alpha_p, E2I_NNZ, e2i_data, e2i_data_pos);'

    if sp_trans == -3.:
        r_slow = '''
        vsSub(NE, h_exc, h_slow_exc, slow_delta);
        cblas_saxpy(NE, dt_tau_slow, slow_delta, 1, h_slow_exc, 1);'''
        slow_init = '''
    static float h_slow_exc[NE] __attribute__((aligned(ALIGNMENT))),
                 slow_delta[NE] __attribute__((aligned(ALIGNMENT)));
    cblas_scopy(NE, h_exc, 1, h_slow_exc, 1);'''
    else:
        r_slow = '''
        vsSub(NE, r_exc, r_slow_exc, slow_delta);
        cblas_saxpy(NE, dt_tau_slow, slow_delta, 1, r_slow_exc, 1);'''
        slow_init = '''
    static float r_slow_exc[NE] __attribute__((aligned(ALIGNMENT))),
                 slow_delta[NE] __attribute__((aligned(ALIGNMENT)));
    cblas_scopy(NE, r_exc, 1, r_slow_exc, 1);'''

    if sp_trans == -4. and sp_type != 'none':
        r_slow += '''
        vsSub(NI, r_inh, r_slow_inh, slow_delta);
        cblas_saxpy(NI, dt_tau_slow_inh, slow_delta, 1, r_slow_inh, 1);'''
        slow_init += '''
    static float r_slow_inh[NI] __attribute__((aligned(ALIGNMENT)));
    const float dt_tau_slow_inh = dt / tau_slow_inh;
    cblas_scopy(NI, r_inh, 1, r_slow_inh, 1);'''
    elif sp_type == 'bcm':
        r_slow += '''
        vsPowx(NI, r_inh, 2, slow_delta);
        vsSub(NI, slow_delta, r_slow_inh2, slow_delta);
        cblas_saxpy(NI, dt_tau_slow_inh, slow_delta, 1, r_slow_inh2, 1);'''
        slow_init += '''
    static float r_slow_inh2[NI] __attribute__((aligned(ALIGNMENT)));
    const float dt_tau_slow_inh = dt / tau_slow_inh;
    cblas_scopy(NI, r_inh, 1, r_slow_inh2, 1);
    vsPowx(NI, r_slow_inh2, 2, r_slow_inh2);'''

    print_slow = ''

    weight_updater = _weight_update(sp_type, i2e_plastic, calibrate_syn, sp_trans,
                                    has_e2e, has_i2i, do_ou_exc, do_ou_inh, weight_update_on_steps,
                                    i2e_binary, do_print_arrays, do_online_report)

    stop_early = ''
    if i2e_plastic and sp_type != 'none':
        stop_early += '''
        for(MKL_INT i = 0; i < I2E_NNZ; ++i)
            if(!isfinite(i2e_data[i]))
            {
                n = n_steps;
                break;
            }'''
    stop_early += '''
        for(MKL_INT i = 0; i < NE; ++i)
            if(!(isfinite(h_exc[i]) && isfinite(in_exc_state[i])))
            {
                n = n_steps;
                break;
            }
        for(MKL_INT i = 0; i < NI; ++i)
            if(!(isfinite(h_inh[i]) && isfinite(in_inh_state[i])))
            {
                n = n_steps;
                break;
            }
        for(MKL_INT i = 0; i < N_PLASTIC; ++i)
            if(!isfinite(plastic_data[i]))
            {
                n = n_steps;
                break;
            }
        if(n == n_steps)
            printf("STOPPING EARLY!\\n");'''

    plasticity_init = weight_updater.plasticity_init
    weight_functions = weight_updater.update_functions
    update_weights = weight_updater.update_weights

    accumulate_inh2exc = ''
    mmap_init = ''
    mmap_cleanup = ''
    if rec_all_plastic:
        if do_mmap:
            mmap_init = '''
    size_t m_page_size = 1000*sysconf(_SC_PAGESIZE), pages_to_flush_size;
    printf("1000*PAGE SIZE %lu \\n", m_page_size);
    int fd_plastic = open(fn_plastic, O_RDWR, (mode_t)0600);
    float *plastic_rec = (float*)mmap(0, fs_plastic, PROT_READ | PROT_WRITE, MAP_SHARED, fd_plastic, 0);
    float *plastic_page_start = plastic_rec;
    if (plastic_rec == MAP_FAILED) {
        close(fd_plastic);
        perror("Error mmapping the plastic-var file");
        exit(EXIT_FAILURE);
    }
    madvise(plastic_rec, fs_plastic, MADV_SEQUENTIAL);'''
            mmap_sync = '''
            pages_to_flush_size = &plastic_rec[frame_n*N_PLASTIC]-plastic_page_start;
            if(pages_to_flush_size > m_page_size)
            {
                pages_to_flush_size = m_page_size;
                msync(plastic_page_start, pages_to_flush_size, MS_INVALIDATE|MS_ASYNC);
                munmap(plastic_page_start, pages_to_flush_size);
                plastic_page_start += pages_to_flush_size;
                printf("e2i PTFS %lu \\n", pages_to_flush_size);
                //madvise(plastic_page_start, pages_to_flush_size, MADV_DONTNEED);
            }'''
            mmap_cleanup = '''
    if (munmap(plastic_rec, fs_plastic) == -1)
        perror("Error un-mmapping the plastic-var file");
    close(fd_plastic);'''
        else:
            mmap_sync = ''

        record_plastic_var = f'''
            cblas_sscal(N_PLASTIC, avg_div, plastic_rec_tmp, 1);
            cblas_scopy(N_PLASTIC, plastic_rec_tmp, 1, &plastic_rec[frame_n*N_PLASTIC], 1);
            {mmap_sync}'''
        if i2e_plastic:
            if do_mmap:
                mmap_init += '''
    int fd_inh2exc = open(fn_inh2exc, O_RDWR, (mode_t)0600);
    float *inh2exc_rec = (float*)mmap(0, fs_inh2exc, PROT_READ | PROT_WRITE, MAP_SHARED, fd_inh2exc, 0);
    float *inh2exc_page_start = inh2exc_rec;
    if (plastic_rec == MAP_FAILED) {
        close(fd_inh2exc);
        perror("Error mmapping the inh2exc file");
        exit(EXIT_FAILURE);
    }
    madvise(inh2exc_rec, fs_inh2exc, MADV_SEQUENTIAL);'''
                mmap_sync = '''
            //msync(&inh2exc_rec[frame_n*I2E_NNZ], I2E_NNZ*sizeof(float), MS_INVALIDATE|MS_ASYNC);
            pages_to_flush_size = &inh2exc_rec[frame_n*I2E_NNZ]-inh2exc_page_start; 
            if(pages_to_flush_size > m_page_size)
            {
                pages_to_flush_size = m_page_size;
                msync(inh2exc_page_start, pages_to_flush_size, MS_INVALIDATE|MS_ASYNC);
                munmap(inh2exc_page_start, pages_to_flush_size);
                inh2exc_page_start += pages_to_flush_size;
                printf("i2e PTFS %lu \\n", pages_to_flush_size);
                //madvise(inh2exc_page_start, pages_to_flush_size, MADV_DONTNEED);
            }'''
                mmap_cleanup = '''
    if (munmap(inh2exc_rec, fs_inh2exc) == -1)
        perror("Error un-mmapping the inh2exc file");
    close(fd_inh2exc);'''
            else:
                mmap_sync = ''
            accumulate_inh2exc = 'vsAdd(I2E_NNZ, i2e_data, inh2exc_rec_tmp, inh2exc_rec_tmp);'
            record_plastic_var += f'''
            cblas_sscal(I2E_NNZ, avg_div, inh2exc_rec_tmp, 1);
            cblas_scopy(I2E_NNZ, inh2exc_rec_tmp, 1, &inh2exc_rec[frame_n*I2E_NNZ], 1);
            {mmap_sync}
            memset(inh2exc_rec_tmp, 0, sizeof(inh2exc_rec_tmp));'''
            plasticity_init += '''
        static float inh2exc_rec_tmp[I2E_NNZ] = {0};'''
    else:
        record_plastic_var = ''
    if sp_type in ('exc2inh', 'aff2in') and sp_trans == -1.:
        record_plastic_var += '''
            vsDiv(NI, num_above_thresh, num_presynaptic, &above_thresh[frame_n*NI]);'''

    do_online_report = 1 if do_online_report else 0
    inner_loop = f'''
    for(MKL_INT n = 0; n < n_steps; ++n)
    {{
        curr_pattern_epoch += (MKL_INT)(input_pattern_epochs[curr_pattern_epoch] < n);
        curr_pattern_index = input_pattern_index[curr_pattern_epoch];
        {update_inputs}
        {update_exc}
        {update_inh}
        vsAdd(NE, r_exc, r_exc_rec_tmp, r_exc_rec_tmp); // Add to running average
        vsAdd(NI, r_inh, r_inh_rec_tmp, r_inh_rec_tmp); 
        vsAdd(NE, h_exc, h_exc_rec_tmp, h_exc_rec_tmp);
        vsAdd(NI, h_inh, h_inh_rec_tmp, h_inh_rec_tmp);
        vsAdd(N_PLASTIC, plastic_data_pos, plastic_rec_tmp, plastic_rec_tmp);
        {accumulate_inh2exc}
        {r_slow}
        {update_weights}

        if((n % steps_per_frame) == compute_on)
        {{
            cblas_sscal(NI, avg_div, h_inh_rec_tmp, 1);
            cblas_scopy(NI, h_inh_rec_tmp, 1, &h_inh_rec[frame_n*NI], 1);
            cblas_sscal(NI, avg_div, r_inh_rec_tmp, 1);
            cblas_scopy(NI, r_inh_rec_tmp, 1, &r_inh_rec[frame_n*NI], 1);

            cblas_sscal(NE, avg_div, h_exc_rec_tmp, 1);
            cblas_scopy(NE, h_exc_rec_tmp, 1, &h_exc_rec[frame_n*NE], 1);
            cblas_sscal(NE, avg_div, r_exc_rec_tmp, 1);
            cblas_scopy(NE, r_exc_rec_tmp, 1, &r_exc_rec[frame_n*NE], 1);

            cblas_saxpy(NE, 1.f, &minus_rho0, 0, r_exc_rec_tmp, 1); // Check error
            vsPowx(NE, r_exc_rec_tmp, 2, r_exc_rec_tmp);
            vslsSSCompute(task_avg_sqr_err, VSL_SS_MEAN, VSL_SS_METHOD_FAST); // VSL_SS_METHOD_1PASS );
            sp_rates[frame_n] = mean_n;

            vslsSSCompute(task_e2i, VSL_SS_MEAN | VSL_SS_2C_MOM, VSL_SS_METHOD_FAST);
            plastic_mean[frame_n] = avg_div * mean_n;
            plastic_var[frame_n] = avg_div*avg_div*variance_n;  // r2m;
            variance_n = 0; r2m = 0;
            {record_plastic_var}
            memset(r_exc_rec_tmp, 0, sizeof(r_exc_rec_tmp));
            memset(r_inh_rec_tmp, 0, sizeof(r_inh_rec_tmp));
            memset(h_exc_rec_tmp, 0, sizeof(h_exc_rec_tmp));
            memset(h_inh_rec_tmp, 0, sizeof(h_inh_rec_tmp));
            memset(plastic_rec_tmp, 0, sizeof(plastic_rec_tmp));

            if({do_online_report})
            {{
                const float one = 1.f;
                float r_exc_mu = cblas_sasum(NE, r_exc, 1) / NE;
                float h_exc_mu = cblas_sdot(NE, &one, 0, h_exc, 1) / NE; // cblas_sasum(NE, h_exc, 1) / NE;
                float r_inh_mu = cblas_sasum(NI, r_inh, 1) / NI;
                MKL_INT maxi_r = cblas_isamax(NE, r_exc, 1);

                printf("r_exc_mu %.2f (h_exc %.2f) RMSE %.1f, max %.1e, (r_inh %g)\\n", 
                       (double)r_exc_mu, (double)h_exc_mu, (double)sqrt(sp_rates[frame_n]), (double)r_exc[maxi_r], (double)r_inh_mu);
                {print_slow}

                pe2("r_exc", r_exc, 1, NE);
                pe2("h_exc", h_exc, 1, NE);
            }}
            frame_n++;
            if(++no_report >= frames_per_report)
            {{
                no_report = 0;
                printf("Computing frame %d / %d (%3.3f)\\n", frame_n, total_frames, (double)100.*frame_n/total_frames);
            }}
            // printf("===================================================================== \\n");
            // eta *= .99;
        }}
        {stop_early}
    }}
    '''

    body = f'''
    static float h_2i[NI]                    __attribute__((aligned(ALIGNMENT))) = {{0}},
                 h_2e[NE]                    __attribute__((aligned(ALIGNMENT))) = {{0}},
                 in_exc_current[NE]          __attribute__((aligned(ALIGNMENT))) = {{0}},
                 in_inh_current[NI]          __attribute__((aligned(ALIGNMENT))) = {{0}},
                 r_exc[NE]                   __attribute__((aligned(ALIGNMENT))) = {{0}},
                 r_inh[NI]                   __attribute__((aligned(ALIGNMENT))) = {{0}},
                 error[NE]                   __attribute__((aligned(ALIGNMENT))) = {{0}},
                 e2i_data_pos[E2I_NNZ]       __attribute__((aligned(ALIGNMENT))) = {{0}},
                 r_exc_rec_tmp[NE]           __attribute__((aligned(ALIGNMENT))) = {{0}},
                 r_inh_rec_tmp[NI]           __attribute__((aligned(ALIGNMENT))) = {{0}},
                 h_exc_rec_tmp[NE]           __attribute__((aligned(ALIGNMENT))) = {{0}},
                 h_inh_rec_tmp[NI]           __attribute__((aligned(ALIGNMENT))) = {{0}},
                 plastic_rec_tmp[N_PLASTIC]  __attribute__((aligned(ALIGNMENT))) = {{0}};
    float *plastic_data = {plastic_var};
    const float minus_rho0 = -rho0, dt_tau_exc = dt / tau_exc, dt_tau_inh = dt / tau_inh, dt_tau_slow = dt / tau_slow;
    const MKL_INT plastic_array_size = N_PLASTIC,
                  n_e = NE, n_i = NI;
    MKL_INT curr_pattern_epoch = 0, curr_pattern_index = input_pattern_index[0], no_report = 0;
    const MKL_INT compute_on = steps_per_frame - 1, 
                  total_frames = n_steps / steps_per_frame, 
                  frames_per_report = n_steps / steps_per_frame / 100;
    int frame_n = 0;
    const float avg_div = 1.f/(float)steps_per_frame;
    mkl_set_num_threads_local({mkl_threads});
    printf("Set number MKL threads %d\\n", {mkl_threads});

    {a2e_init}
    {a2i_init}
    {make_e2i_pos}
    {plasticity_init}
    {ou_init}
    {slow_init}
    {mmap_init}
    printf("Target rate is %g Hz\\n", (double)rho0);

    VSLSSTaskPtr task_avg_sqr_err, task_e2i; /* SS task descriptor */
    float mean_n = 0, variance_n = 0, r2m = 0; /* Arrays for estimates */
    float* w = 0; /* Null pointer to array of weights, default weight equal to one will be used in the computation */
    MKL_INT mean_dim = 1, xstorage = VSL_SS_MATRIX_STORAGE_ROWS;

    vslsSSNewTask( &task_e2i, &mean_dim, &plastic_array_size, &xstorage, plastic_rec_tmp, w, 0 );
    vslsSSEditTask( task_e2i, VSL_SS_ED_MEAN, &mean_n );
    vslsSSEditTask( task_e2i, VSL_SS_ED_2C_MOM, &variance_n );
    vslsSSEditMoments(task_e2i, &mean_n, &r2m, 0, 0, &variance_n, 0, 0);

    vslsSSNewTask( &task_avg_sqr_err, &mean_dim, &n_e, &xstorage, r_exc_rec_tmp, w, 0 );
    vslsSSEditTask( task_avg_sqr_err, VSL_SS_ED_MEAN, &mean_n );

    rate_nonlinearity(f_max, alpha_r, NE, h_exc, r_exc, NULL, NULL);
    rate_nonlinearity(f_max, alpha_r, NI, h_inh, r_inh, inh_thresh, inh_gain_pos);

    printf("Started C loop \\n");
    {inner_loop}
    printf("Finished C loop\\n");

    vslSSDeleteTask( &task_avg_sqr_err );
    vslSSDeleteTask( &task_e2i );
    {ou_cleanup}
    {mmap_cleanup}
    '''

    p_types_names = [('const MKL_INT', 'n_steps'),
                     ('const float', 'dt'),
                     ('const float', 'tau_exc'),
                     ('const float', 'tau_inh'),
                     ('const float', 'tau_slow'),
                     ('const MKL_INT', 'steps_per_frame'),
                     ('const MKL_UINT', 'seed'),
                     ('const float', 'rho0'),
                     ('const MKL_INT', 'n_calibrate'),
                     ('const float', 'alpha_r'),
                     ('const float', 'alpha_p'),
                     ('const float', 'f_max'),
                     ('const MKL_INT *', 'e2i_indptr'),
                     ('const MKL_INT *', 'e2i_indices'),
                     ('const MKL_INT *', 'i2e_indptr'),
                     ('const MKL_INT *', 'i2e_indices'),
                     ('const MKL_INT', 'plasticity_on'),
                     ('const MKL_INT', 'plasticity_off'),
                     ('const float *', 'in_exc_mean'),
                     ('const float *', 'in_inh_mean'),
                     ('const float *', 'in_exc_patterns'),
                     ('float *', 'in_exc_state'),
                     ('float *', 'in_inh_state'),
                     ('const MKL_INT *', 'input_pattern_epochs'),
                     ('const MKL_INT *', 'input_pattern_index'),
                     ('float *', 'h_exc'),
                     ('float *', 'h_inh'),
                     ('float *', 'r_exc_rec'),
                     ('float *', 'r_inh_rec'),
                     ('float *', 'h_exc_rec'),
                     ('float *', 'h_inh_rec'),
                     ('float *', 'plastic_mean'),
                     ('float *', 'plastic_var'),
                     ('float *', 'sp_rates'),
                     ('const float', 'rho_calibrate'),
                     ('const float', 'calibration_eta')
                     ]
    if has_e2e:
        p_types_names += [('const float *', 'e2e_data'),
                          ('const MKL_INT *', 'e2e_indptr'),
                          ('const MKL_INT *', 'e2e_indices')]
    if has_i2i:
        p_types_names += [('const float *', 'i2i_data'),
                          ('const MKL_INT *', 'i2i_indptr'),
                          ('const MKL_INT *', 'i2i_indices')]
    if do_aff_inh:
        p_types_names += [('const float *', 'in_inh_patterns'),
                          ('const MKL_INT *', 'a2i_indptr'),
                          ('const MKL_INT *', 'a2i_indices'), ]
        if sp_type == 'aff2inh':
            p_types_names += [('float *', 'a2i_data'),
                              ('const MKL_INT *', 'a2i_data_i'),
                              ('const MKL_INT *', 'a2i_data_j'), ]
        elif 'a2i' in calibrate_syn:
            p_types_names += [('float *', 'a2i_data')]
        else:
            p_types_names += [('const float *', 'a2i_data')]

    if sp_type == 'inh_gain':
        p_types_names += [('const float *', 'e2i_data'),
                          ('float *', 'inh_gain')]
    elif sp_type == 'inh_thresh':
        p_types_names += [('const float *', 'e2i_data'),
                          ('float *', 'inh_thresh')]
    elif sp_type == 'nonhebb':
        p_types_names += [('float *', 'e2i_data'),
                          ('const MKL_INT *', 'e2i_data_j'), ]
    elif sp_type in ('exc2inh', 'hebbian', 'antihebb', 'bcm'):
        p_types_names += [('float *', 'e2i_data'),
                          ('const MKL_INT *', 'e2i_data_i'),
                          ('const MKL_INT *', 'e2i_data_j'), ]
    elif 'e2i' in calibrate_syn:
        p_types_names += [('float *', 'e2i_data')]
    else:
        p_types_names += [('const float *', 'e2i_data')]

    if sp_type != 'none' or 'e2i' in calibrate_syn:
        p_types_names += [('const float', 'j_max')]

    if sp_type == 'antihebb':
        p_types_names += [('const float', 'r_antihebb')]

    if i2e_plastic:
        p_types_names += [('const float', 'eta_i2e'),
                          ('float *', 'i2e_data'),
                          ('const MKL_INT *', 'i2e_data_i'),
                          ('const MKL_INT *', 'i2e_data_j'),
                          ('const float', 'j_i2e_max')]
    elif 'i2e' in calibrate_syn:
        p_types_names += [('float *', 'i2e_data'),
                          ('const float', 'j_i2e_max')]
    else:
        p_types_names += [('const float *', 'i2e_data')]

    if sp_type in ('exc2inh', 'aff2inh') and sp_trans == -1.:
        p_types_names += [('const float', 'x2i_thresh'),
                          ('float *', 'above_thresh')]
    if sp_type != 'none':
        p_types_names += [('const float', 'eta')]
        if has_k:
            p_types_names += [('const float *', 'k')]
        elif sp_trans == -2:
            p_types_names += [('float *', 'i2e_data_normed')]
            if i2e_plastic:
                p_types_names += [('const float', 'i2e_alpha'),
                                  ('const float', 'i2e_thresh')]
        elif sp_trans == -3.:
            p_types_names += [('const float', 'bls_c'),
                              ('const float', 'bls_tau'),
                              ('const float', 'bls_alpha_lb'),
                              ('float *', 'eta_rec'),
                              ('float *', 'error_rec'),
                              ('float *', 'bls_t_rec')]
        elif sp_trans == -4.:
            p_types_names += [('const float', 'rho0_inh'),
                              ('const float', 'tau_slow_inh')]

    if weight_update_on_steps:
        p_types_names += [('const MKL_INT *', 'update_weights_n')]

    else:
        p_types_names += [('const MKL_INT', 'update_weights_n')]

    if rec_all_plastic:
        if do_mmap:
            p_types_names += [('char *', 'fn_plastic'),
                              ('size_t', 'fs_plastic')]
            if i2e_plastic:
                p_types_names += [('char*', 'fn_inh2exc'),
                                  ('size_t', 'fs_inh2exc')]
        else:
            p_types_names += [('float*', 'plastic_rec')]
            if i2e_plastic:
                p_types_names += [('float*', 'inh2exc_rec')]
    if do_ou_exc:
        p_types_names += [('const float', 'in_ou_exc_tau'),
                          ('const float', 'in_ou_exc_sigma')]
    if do_ou_inh:
        p_types_names += [('const float', 'in_ou_inh_tau'),
                          ('const float', 'in_ou_inh_sigma')]

    param_list = ', '.join(' '.join((t, n)) for t, n in p_types_names)
    main_declaration = f'void inner_impl({param_list})'
    main_definition = f'''
{weight_functions}
{main_declaration}
{{
    {body}
}}'''
    return function_definitions(main_declaration, main_definition, p_types_names)


def inner_c(n_e: int, n_i: int, sp_type: InterneuronPlasticity, sp_subtype: PlasticitySubtype,
            i2e_binary: bool, calibrate_syn: str, build_path: str,
            do_print_arrays: bool, rec_all_plastic: bool, do_mmap: bool, **kwargs) -> Dict[str, Union[str, Dict]]:
    from utils import allocate_aligned, reallocate_aligned
    from cffi import FFI
    import os

    sp_type = sp_type.name.lower()

    if sp_subtype in (PlasticitySubtype.DIFFUSIVE, PlasticitySubtype.LOCAL):
        sp_trans = dtype(kwargs.pop("diffusion_dist", 0.))
    else:
        sp_trans = dtype(sp_subtype.value)

    assert sp_type in ('none', 'hebbian', 'nonhebb', 'antihebb', 'exc2inh', 'inh_gain', 'inh_thresh', 'bcm', 'aff2inh')

    if sp_type in ('nonhebb', 'hebbian', 'antihebb') and sp_trans > 0.:
        raise Exception(f'{sp_type} plasticity is always local, transmission value {sp_trans} makes no sense')
    has_k: bool = sp_trans > -1. and sp_type in ('exc2inh', 'inh_thresh', 'inh_gain', 'aff2inh')
    i2e_plastic: bool = kwargs.pop('i2e_plastic') or sp_type == 'inh2exc'

    do_ou_exc: bool = 'in_ou_exc_sigma' in kwargs and kwargs['in_ou_exc_sigma'] > 0. and 'in_ou_exc_tau' in kwargs
    do_ou_inh: bool = 'in_ou_inh_sigma' in kwargs and kwargs['in_ou_inh_sigma'] > 0. and 'in_ou_inh_tau' in kwargs
    if not do_ou_exc:
        kwargs.pop('in_ou_exc_tau', None)
        kwargs.pop('in_ou_exc_sigma', None)
    if not do_ou_inh:
        kwargs.pop('in_ou_inh_tau', None)
        kwargs.pop('in_ou_inh_sigma', None)

    alpha_r: float = kwargs['alpha_r']
    alpha_p: float = kwargs['alpha_p']

    n_a2i: int = kwargs['in_inh_patterns'].shape[1] if 'in_inh_patterns' in kwargs else 0

    do_aff_inh: bool = 'a2i_data' in kwargs
    a2i_nnz: int = kwargs['a2i_data'].size if do_aff_inh else 0
    e2i_nnz: int = kwargs['e2i_data'].size
    i2e_nnz: int = kwargs['i2e_data'].size
    has_i2i: bool = 'i2i_data' in kwargs
    has_e2e: bool = 'e2e_data' in kwargs

    # These are used for both recording the mean & variance of the plastic variable, as well as
    # when recording all values (when rec_all_plastic is true).
    if sp_type in ('inh_gain', 'inh_thresh'):
        plastic_var = sp_type
        n_plastic_str = 'NI'
        n_plastic = n_i
    elif sp_type in ('exc2inh', 'nonhebb', 'hebbian', 'antihebb', 'bcm'):
        plastic_var = 'e2i_data'
        n_plastic_str = 'E2I_NNZ'
        n_plastic = e2i_nnz
    elif sp_type == 'aff2inh':
        plastic_var = 'a2i_data'
        n_plastic_str = 'A2I_NNZ'
        n_plastic = a2i_nnz
    elif sp_type == 'none':
        if i2e_plastic and not rec_all_plastic:
            # If only i2e is plastic, track its mean and variance, when not recording all values.
            plastic_var = 'i2e_data'
            n_plastic_str = 'I2E_NNZ'
            n_plastic = i2e_nnz
        else:
            # This is just so there is some valid data to record, and we can track calibration
            plastic_var = 'e2i_data'
            n_plastic_str = '1'
            n_plastic = 1
    else:
        raise Exception(f'Plasticity type not implemented ({sp_type})')

    weight_update_on_steps = isinstance(kwargs['update_weights_n'], np.ndarray)

    c_lib = _prepare_inner_c(n_e, n_i, e2i_nnz, i2e_nnz, a2i_nnz, alpha_r, alpha_p,
                             sp_type, sp_trans, plastic_var, weight_update_on_steps, i2e_binary,
                             n_plastic_str,
                             has_k, i2e_plastic, calibrate_syn,
                             do_ou_exc, do_ou_inh, do_aff_inh, n_a2i,
                             has_e2e, has_i2i,
                             build_path, do_print_arrays,
                             rec_all_plastic, do_mmap)

    ffi = FFI()
    with open(c_lib.h_file_path, 'r') as f:
        ffi.cdef(f.read())
    lib = ffi.dlopen(c_lib.so_file_path)

    n_steps = kwargs['n_steps']
    steps_per_frame = kwargs['steps_per_frame']
    rec_n = n_steps // steps_per_frame

    align = 32
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            needs_realignment = v.ctypes.data % align
            if needs_realignment:
                # raise Exception(f'{k} is not aligned on a {align}-byte boundary, use allocate_aligned')
                kwargs[k] = reallocate_aligned(v)

    rec = dict(r_exc_rec=allocate_aligned((rec_n, n_e), dtype=dtype),
               r_inh_rec=allocate_aligned((rec_n, n_i), dtype=dtype),
               h_exc_rec=allocate_aligned((rec_n, n_e), dtype=dtype),
               h_inh_rec=allocate_aligned((rec_n, n_i), dtype=dtype),
               sp_rates=allocate_aligned(rec_n, dtype=dtype),
               plastic_mean=allocate_aligned(rec_n, dtype=dtype),
               plastic_var=allocate_aligned(rec_n, dtype=dtype))

    if sp_trans == -3. and sp_type != 'none':
        plasticity_on = kwargs['plasticity_on']
        plasticity_off = kwargs['plasticity_off']
        update_weights_n = kwargs['update_weights_n']
        n_plastic_steps = max(1, (n_steps - (plasticity_on + (n_steps - plasticity_off))) // update_weights_n)
        rec['eta_rec'] = allocate_aligned(n_plastic_steps, dtype=dtype)
        rec['error_rec'] = allocate_aligned(n_plastic_steps, dtype=dtype)
        rec['bls_t_rec'] = allocate_aligned(n_plastic_steps, dtype=dtype)
    elif sp_trans == -1. and sp_type in ('exc2inh', 'aff2inh'):
        rec['above_thresh'] = allocate_aligned((rec_n, n_i), dtype=dtype)

    fn_i2e = os.path.join(build_path, 'inh2exc_rec')
    fn_plastic = os.path.join(build_path, 'plastic_rec')
    if rec_all_plastic:
        if do_mmap:
            f_plastic = np.memmap(fn_plastic, dtype=dtype, mode='w+', shape=(rec_n, n_plastic))
            print(f'created mmap at {fn_plastic}')
            rec['fn_plastic'] = fn_plastic.encode()
            rec['fs_plastic'] = np.uintp(f_plastic.size * f_plastic.dtype.itemsize)
            del f_plastic
        else:
            rec['plastic_rec'] = allocate_aligned((rec_n, n_plastic), dtype=dtype)

        if i2e_plastic:
            if do_mmap:
                f_i2e = np.memmap(fn_i2e, dtype=dtype, mode='w+', shape=(rec_n, i2e_nnz))
                print(f'created mmap at {fn_i2e}')
                rec['fn_inh2exc'] = fn_i2e.encode()
                rec['fs_inh2exc'] = np.uintp(f_i2e.size * f_i2e.dtype.itemsize)
                del f_i2e
            else:
                rec['inh2exc_rec'] = allocate_aligned((rec_n, i2e_nnz), dtype=dtype)

    kwargs.update(rec)

    p_names = [n for t, n in c_lib.p_types_names]
    missing_a = set(p_names).difference(kwargs.keys())
    missing_b = set(kwargs.keys()).difference(p_names)
    assert 0 == len(missing_a), f'Expected parameters missing: {missing_a}'
    assert 0 == len(missing_b), f'Too many parameters provided: {missing_b}'

    args = []
    for p_type, p_name in c_lib.p_types_names:
        pv = kwargs[p_name]
        if 'char' in p_type:
            pv = ffi.cast(p_type, ffi.from_buffer(pv))
        elif '*' in p_type:
            assert isinstance(pv, np.ndarray), f"Expected array for {p_name}"
            pv_dtype = pv.dtype
            err_str = f"Expected {p_type} for {p_name} got {pv_dtype} *"
            assert (('char' in p_type and pv_dtype == np.uint16) or
                    ('float' in p_type and pv_dtype == np.float32) or
                    ('double' in p_type and pv_dtype == np.float64) or
                    ('MKL_UINT' in p_type and pv_dtype in (np.uint32, np.uint64)) or
                    ('MKL_INT' in p_type and pv_dtype in (np.int32, np.int64))), err_str
            pv = ffi.cast(p_type, ffi.from_buffer(pv))
        else:
            type_pv = type(pv)
            types = (int, np.uintp, dtype)
            assert type_pv in types, f"Parameter {p_name} {type_pv} {dtype}"

        args.append(pv)

    try:
        lib.inner_impl(*args)
    except KeyboardInterrupt:
        print('Simulation was cancelled with keyboard interrupt.')

    data = {'code': c_lib.h_c_source,
            'rates': {},
            'sparseness': {'rates': rec['sp_rates']},
            'plastic': {'mean': rec['plastic_mean'],
                        'var': rec['plastic_var']}}

    for n in ('exc', 'inh'):
        for t in ('h', 'r'):
            rhs = f'{t}_{n}_rec'
            lhs = n if t == 'r' else f'h_{n}'
            try:
                data['rates'][lhs] = rec[rhs].T
            except ValueError as e:
                print(e)
                print(rhs, rec[rhs].shape)
                print('=' * 80)

    if sp_trans == -3. and sp_type != 'none':
        data['eta'] = rec['eta_rec']
        data['error'] = rec['error_rec']
        data['bls_t'] = rec['bls_t_rec']
    elif sp_trans == -1. and sp_type in ('exc2inh', 'aff2inh'):
        data['above_thresh'] = rec['above_thresh'].T
    if rec_all_plastic:
        if do_mmap:
            from .utils import mmap_array
            data['plastic']['all'] = mmap_array('MMAP', dtype, (rec_n, n_plastic), fn_plastic, True)
            if i2e_plastic:
                data['plastic']['inh2exc'] = mmap_array('MMAP', dtype, (rec_n, i2e_nnz), fn_i2e, True)
        else:
            data['plastic']['all'] = rec['plastic_rec'].T
            if i2e_plastic:
                data['plastic']['inh2exc'] = rec['inh2exc_rec'].T

    return data
