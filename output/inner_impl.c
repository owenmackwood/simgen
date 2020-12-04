

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

#define NE 512
#define NI 128
#define E2I_NNZ 6625
#define I2E_NNZ 6594
#define A2I_NNZ 0
#define N_PLASTIC E2I_NNZ
#define N_A2I 0
#define ALIGNMENT 64
const char *no_trans = "N", *do_trans = "T";



#define PRINT_ARRAYS
#ifdef PRINT_ARRAYS
    #define pe2(s, a, m, n) print_2d_float(s, a, m, n)
    #define pia(s, a, n) print_1d_int(s, a, n)
    #define psa(s, a, nnz, m, n, columns, row_ptr) print_sparse(s, a, nnz, m, n, columns, row_ptr)
#else
    #define pe2(s, a, m, n)
    #define pia(s, a, n)
    #define psa(s, a, nnz, m, n, columns, row_ptr)
#endif

void print_2d_float(char* s, const float* a, int m, int n)
{
    printf("%s  = np.loadtxt(StringIO(''' \n", s);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++) printf("%.6a ", (double)a[i*n + j]);
        printf("\n");
    }
    printf("'''), dtype=np.float32)\n");
}

void print_1d_int(char* s, const int* a, int n)
{
    printf("%s  = np.loadtxt(StringIO(''' \n", s);
    for(int i = 0; i < n; i++)
        printf("%d ", a[i]);
    printf("'''), dtype=np.int32)\n");
}

void print_sparse(char* s, const float* a, int nnz, int m, int n, const int *columns, const int *row_ptr)
{
    if(1)
    {
        printf("%s  = np.loadtxt(StringIO(''' \n", s);
        for(int i = 0; i < m; i++)
        { // For each row
            int start = row_ptr[i], end = row_ptr[i+1];
            if(start == end)
                for(int j = 0; j < n; ++j) printf("%.6a ", .0);
            else
            {
                int col = 0;
                for(int j = start; j < end; ++j)
                {
                    while(col++ < columns[j])
                        printf("%.6a ", .0);
                    printf("%.6a ", (double)a[j]);
                }
                while(col++ < n)
                    printf("%.6a ", .0);
            }
            printf("\n");
        }
        printf("'''), dtype=np.float32)\n");
    }
    else
    {
        char data_str[160], indices_str[160], indptr_str[160];
        sprintf(data_str, "%s_data", s);
        print_2d_float(data_str, a, 1, nnz);
        sprintf(indices_str, "%s_indices", s);
        print_1d_int(indices_str, columns, nnz);
        sprintf(indptr_str, "%s_indptr", s);
        print_1d_int(indptr_str, row_ptr, m+1);
        printf("%s = csr_matrix((%s, %s, %s), shape=(%d, %d))\n", s, data_str, indices_str, indptr_str, m, n);

    }
}


typedef enum {ROW_WISE, COL_WISE} axis_dir;
void sum_axis(axis_dir nt, const float* a, const int m_rows, const int n_cols,
              const int *columns, const int *row_ptr, float* sums)
{
    const float one = 1.f;
    a = __builtin_assume_aligned(a, ALIGNMENT);
    columns = __builtin_assume_aligned(columns, ALIGNMENT);
    row_ptr = __builtin_assume_aligned(row_ptr, ALIGNMENT);
    sums = __builtin_assume_aligned(sums, ALIGNMENT);
    if(nt == ROW_WISE)
    {
        for(int i = 0; i < m_rows; i++)
        { // For each row
            const int start = row_ptr[i], end = row_ptr[i+1];
            sums[i] = cblas_sdot(end-start, &one, 0, &a[start], 1); // if end-start <= 0 returns 0
            //const float row_sum = cblas_sasum(end-start, &a[start], 1); can't use, computes sum of magnitudes
        }
    }
    else
    {
        memset(sums, 0, n_cols * sizeof(float));
        for(int i = 0; i < m_rows; i++)
        { // For each row
            int start = row_ptr[i], end = row_ptr[i+1];
            for(int j = start; j < end; ++j)
                sums[columns[j]] += a[j];
        }
    }
}

void div_by_sums(axis_dir nt, const float* a, const int nnz, const int m_rows, const int n_cols,
                    const int *columns, const int *row_ptr, const float* sums, float* b)
{
    const float zero = 1e-9f;
    a = __builtin_assume_aligned(a, ALIGNMENT);
    columns = __builtin_assume_aligned(columns, ALIGNMENT);
    row_ptr = __builtin_assume_aligned(row_ptr, ALIGNMENT);
    sums = __builtin_assume_aligned(sums, ALIGNMENT);
    b = __builtin_assume_aligned(b, ALIGNMENT);
    cblas_scopy(nnz, a, 1, b, 1); // Copy data from a -> b

    const int len_sums = (nt == ROW_WISE) ? m_rows : n_cols;

    float* _inv_sums = (float*)mkl_calloc(len_sums, sizeof(float), ALIGNMENT);
    for(int i = 0; i < len_sums; ++i)
        _inv_sums[i] = (abs(sums[i]) < zero) ? 1.f : sums[i];
    vsInv(len_sums, _inv_sums, _inv_sums);

    if(nt == ROW_WISE)
    {
        for(int i = 0; i < m_rows; i++)
        { // For each row
            const int start = row_ptr[i], end = row_ptr[i+1];
            cblas_sscal(end-start, _inv_sums[i], &b[start], 1);
        }
    }
    else
    {
        for(int i = 0; i < m_rows; i++)
        { // For each row
            const int start = row_ptr[i], end = row_ptr[i+1];
            for(int j = start; j < end; ++j)
                b[j] *= _inv_sums[columns[j]];
        }
    }
    mkl_free(_inv_sums);
}

void normalize_sparse(axis_dir nt, const float* a, const int nnz, const int m_rows, const int n_cols,
                      const int *columns, const int *row_ptr, float* b)
{
    float* sums = (float*)mkl_calloc((nt == ROW_WISE)?m_rows:n_cols, sizeof(float), ALIGNMENT); // alloc and init to zero
    sum_axis(nt, a, m_rows, n_cols, columns, row_ptr, sums);
    div_by_sums(nt, a, nnz, m_rows, n_cols, columns, row_ptr, sums, b);
    mkl_free(sums);
}

typedef enum {GAIN, THRESH, OTHER} derivative_wrt;
void soft_plus_with_max(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain)
{
    // y = log((1 + exp(alpha*x)) /    (1 + exp(alpha*(x - max_y)))) / alpha
    //   = (log(1 + exp(alpha*x)) - log(1 + exp(alpha*(x - max_y)))) / alpha

    assert(thresh == NULL && gain == NULL);
    float* tmp = (float*)mkl_calloc(n, sizeof(float), ALIGNMENT);

    x = __builtin_assume_aligned(x, ALIGNMENT);
    tmp = __builtin_assume_aligned(tmp, ALIGNMENT);
    y = __builtin_assume_aligned(y, ALIGNMENT);

    for(int i = 0; i < n; ++i)
        y[i] = alpha * x[i];
    vsExp(n, y, y);
    vsLog1p(n, y, y);
    for(int i = 0; i < n; ++i)
        y[i] = isfinite(y[i]) ? y[i] : alpha*x[i];

    for(int i = 0; i < n; ++i)
        tmp[i] = alpha*(x[i] - max_y);
    vsExp(n, tmp, tmp);
    vsLog1p(n, tmp, tmp);
    for(int i = 0; i < n; ++i)
        tmp[i] = isfinite(tmp[i]) ? tmp[i] : alpha*(x[i] - max_y);

    vsSub(n, y, tmp, y);
    for(int i = 0; i < n; ++i)
        y[i] /= alpha;

    for(int i = 0; i < n; ++i)
        assert(isfinite(y[i]));

    mkl_free(tmp);
}

void soft_plus_with_max_derivative(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain, const derivative_wrt wrt)
{
    // yp = 1/(exp(alpha*(x-max_y))+1) - 1/(exp(alpha*x)+1)

    assert(wrt == OTHER && thresh == NULL && gain == NULL);
    float* tmp = (float*)mkl_calloc(n, sizeof(float), ALIGNMENT);

    x = __builtin_assume_aligned(x, ALIGNMENT);
    tmp = __builtin_assume_aligned(tmp, ALIGNMENT);
    y = __builtin_assume_aligned(y, ALIGNMENT);

    for(int i = 0; i < n; ++i)
        y[i] = alpha * x[i];
    vsExp(n, y, y);
    for(int i = 0; i < n; ++i)
        y[i] += 1.f;
    vsInv(n, y, y);

    for(int i = 0; i < n; ++i)
        tmp[i] = alpha * (x[i] - max_y);
    vsExp(n, tmp, tmp);
    for(int i = 0; i < n; ++i)
        tmp[i] += 1.f;
    vsInv(n, tmp, tmp);

    vsSub(n, tmp, y, y);

    for(int i = 0; i < n; ++i)
        assert(isfinite(y[i]));

    mkl_free(tmp);
}
void plastic_nonlinearity(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y)
{
    soft_plus_with_max(max_y, alpha, n, x, y, NULL, NULL);
}                               
void plastic_nonlinearity_deriv(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y)
{
    soft_plus_with_max_derivative(max_y, alpha, n, x, y, NULL, NULL, OTHER);
} 
void rate_nonlinearity(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain)
{
    soft_plus_with_max(max_y, alpha, n, x, y, thresh, gain);
}                               
void rate_nonlinearity_deriv(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain, const derivative_wrt wrt)
{
    soft_plus_with_max_derivative(max_y, alpha, n, x, y, thresh, gain, wrt);
}                               


void update_plastic(float * error, const float eta, const float j_max, float * plastic_data, float * plastic_delta, const float alpha_r, float * h_inh, float * h_slow_exc, float * in_exc_state, float * in_inh_state, float * h_exc, const MKL_INT n_e, const MKL_INT n_i, const float * i2e_data, const MKL_INT * i2e_indptr, const MKL_INT * i2e_indices, const float * e2i_data_pos, const MKL_INT * e2i_indptr, const MKL_INT * e2i_indices, const float bls_c, const float bls_tau, const float bls_alpha_lb, float * eta_rec, float * error_rec, float * bls_t_rec, const float * e2e_data, const MKL_INT * e2e_indptr, const MKL_INT * e2e_indices, const float * i2i_data, const MKL_INT * i2i_indptr, const MKL_INT * i2i_indices, float * plastic_data_pos, const MKL_INT * x2i_data_i, const MKL_INT * x2i_data_j, const float* r_pre, const float alpha_p, const float * inh_thresh, const float * inh_gain_pos)
{
    
    static float plastic_logistic[N_PLASTIC] __attribute__((aligned(ALIGNMENT))) = {0},
                 dri_dhi[NI] __attribute__((aligned(ALIGNMENT))) = {0},
                 dre_dhe[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 w_eff[NE*NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 sqr_error[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 error_tmp[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 grad_buff[NE*NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 plastic_data_prev[N_PLASTIC] __attribute__((aligned(ALIGNMENT))) = {0},
                 h_exc_prev[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 h_inh_prev[NI] __attribute__((aligned(ALIGNMENT))) = {0},
                 h_slow_exc_prev[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 bls_p[N_PLASTIC] __attribute__((aligned(ALIGNMENT))) = {0},
                 in_exc_state_prev[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                 ii_eff[NI*NI] __attribute__((aligned(ALIGNMENT))) = {0},
                 w_eff_tmp[NE*NE] __attribute__((aligned(ALIGNMENT))) = {0};
    bool error_increased = false;
    static float total_error_prev = INFINITY;
    float one_f = 1.f, zero_f = 0.f;
    float bls_alpha = eta, bls_t = 0.f;
    static MKL_INT ipiv[NE] __attribute__((aligned(ALIGNMENT))) = {0},
                   eta_n = 0, info;
    const char *matdescra = "G__C_"; 
    static MKL_INT ipiv_ii[NI] __attribute__((aligned(ALIGNMENT))) = {0};
    
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
    }
    if(!error_increased)
    {
        
        // If the error decreased, restore step size and save state before recomputing gradient
        bls_alpha = eta;
        total_error_prev = total_error;

        // Store current state before update
        memcpy(plastic_data_prev, plastic_data, sizeof(plastic_data_prev));
        memcpy(h_slow_exc_prev, h_slow_exc, sizeof(h_slow_exc_prev)); 
        memcpy(h_exc_prev, h_exc, sizeof(h_exc_prev));
        memcpy(h_inh_prev, h_inh, sizeof(h_inh_prev));
        memcpy(in_exc_state_prev, in_exc_state, sizeof(in_exc_state_prev));
          
        
        // compute w_eff = (1+w_i2e dri_dhi w_e2i dre_dhe)
        // or w_eff = (1+w_i2e ii_eff dri_dhi w_e2i dre_dhe + w_e2e dre_dhe)
        memset(grad_buff, 0, sizeof(grad_buff));

        rate_nonlinearity_deriv(f_max, alpha_r, NI, h_inh, dri_dhi, inh_thresh, inh_gain_pos, OTHER);  // Fix this line
        pe2("dri_dhi", dri_dhi, 1, NI);
        
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
        LAPACKE_sgetrf(LAPACK_ROW_MAJOR, NI, NI, ii_eff, NI, ipiv_ii);  

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
                printf( "The diagonal element of the triangular factor of A,\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "the solution could not be computed.\n" );
                exit( 1 );
        }

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
        vsSub(NE*NE, w_eff, w_eff_tmp, w_eff);

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
        if( info > 0 ) {
                printf( "The diagonal element of the triangular factor of A,\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "the solution could not be computed.\n" );
                exit( 1 );
        }

        pe2("error_rec", error, 1, NE);

        // error now contains the error evaluated through the recurrent connections.
        // We now back-trace that error through the inhibitory connectivity. [error*(1+ww diag)^-1] * wei
        mkl_cspblas_scsrgemv(do_trans, &n_e, i2e_data, i2e_indptr, i2e_indices, error, error_tmp);
        cblas_scopy(NI, error_tmp, 1, error, 1);
        
        info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', // test change 'T', 
                       NI, // n: The order of A; the number of rows in B(n≥ 0).
                       1, // nrhs: Number of right hand sides
                       ii_eff, NI, // lda: The leading dimension of a; lda>=max(1, n)
                       ipiv_ii, error, 1 // ldb: The leading dimension of b; ldb>=nrhs for row major layout.
                       );
        if( info > 0 ) {
                printf( "The diagonal element of the triangular factor of A,\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "the solution could not be computed.\n" );
                exit( 1 );
        }

        pe2("backprop_err", error, 1, NI);  
        
        // dri_dhi already calculated during error propagation
        // Prefer synapses onto active interneurons.
        vsMul(NI, dri_dhi, error, error);
        // Prefer synapses driven by cells with high rates
        vsPackV(N_PLASTIC, r_pre, x2i_data_j, plastic_logistic); 
        vsPackV(N_PLASTIC, error, x2i_data_i, plastic_delta); // Take error into weight change array.
        vsMul(N_PLASTIC, plastic_logistic, plastic_delta, plastic_delta); // Multiply error by pre-rates.
        // derivative of e2i wrt v, enforces non-negativity.
        plastic_nonlinearity_deriv(j_max, alpha_p, N_PLASTIC, plastic_data, plastic_logistic);

        vsMul(N_PLASTIC, plastic_logistic, plastic_delta, plastic_delta);
                // Store the normalized gradient in bls_p, and compute the update threshold.
                cblas_scopy(N_PLASTIC, plastic_delta, 1, bls_p, 1);
                cblas_sscal(N_PLASTIC, 1.f / cblas_snrm2(N_PLASTIC, bls_p, 1), bls_p, 1); // Normalize gradient
                bls_t = bls_c * cblas_sdot(N_PLASTIC, bls_p, 1, plastic_delta, 1); // Compute threshold t for Armijo-Goldstein
                cblas_scopy(N_PLASTIC, bls_p, 1, plastic_delta, 1);
                cblas_sscal(N_PLASTIC, bls_alpha, plastic_delta, 1);
                
        
        vsAdd(N_PLASTIC, plastic_delta, plastic_data, plastic_data);
        plastic_nonlinearity(j_max, alpha_p, N_PLASTIC, plastic_data, plastic_data_pos);
    }
    else
    {
        
        // If the error increased, undo the change and decrease step size.
        bls_alpha *= bls_tau;

        // Compute a new delta and apply
        cblas_scopy(N_PLASTIC, bls_p, 1, plastic_delta, 1);
        cblas_sscal(N_PLASTIC, bls_alpha, plastic_delta, 1);
        memcpy(plastic_data, plastic_data_prev, sizeof(plastic_data_prev)); 
        
        vsAdd(N_PLASTIC, plastic_delta, plastic_data, plastic_data);
        plastic_nonlinearity(j_max, alpha_p, N_PLASTIC, plastic_data, plastic_data_pos);

        // Restore remaining state
        memcpy(h_slow_exc, h_slow_exc_prev, sizeof(h_slow_exc_prev));
        memcpy(h_exc, h_exc_prev, sizeof(h_exc_prev));
        memcpy(h_inh, h_inh_prev, sizeof(h_inh_prev));
        memcpy(in_exc_state, in_exc_state_prev, sizeof(in_exc_state_prev));
          
    }
}
void inner_impl(const MKL_INT n_steps, const float dt, const float tau_exc, const float tau_inh, const float tau_slow, const MKL_INT steps_per_frame, const MKL_UINT seed, const float rho0, const MKL_INT n_calibrate, const float alpha_r, const float alpha_p, const float f_max, const MKL_INT * e2i_indptr, const MKL_INT * e2i_indices, const MKL_INT * i2e_indptr, const MKL_INT * i2e_indices, const MKL_INT plasticity_on, const MKL_INT plasticity_off, const float * in_exc_mean, const float * in_inh_mean, const float * in_exc_patterns, float * in_exc_state, float * in_inh_state, const MKL_INT * input_pattern_epochs, const MKL_INT * input_pattern_index, float * h_exc, float * h_inh, float * r_exc_rec, float * r_inh_rec, float * h_exc_rec, float * h_inh_rec, float * plastic_mean, float * plastic_var, float * sp_rates, const float rho_calibrate, const float calibration_eta, const float * e2e_data, const MKL_INT * e2e_indptr, const MKL_INT * e2e_indices, const float * i2i_data, const MKL_INT * i2i_indptr, const MKL_INT * i2i_indices, float * e2i_data, const MKL_INT * e2i_data_i, const MKL_INT * e2i_data_j, const float j_max, const float eta_i2e, float * i2e_data, const MKL_INT * i2e_data_i, const MKL_INT * i2e_data_j, const float j_i2e_max, const float eta, const float bls_c, const float bls_tau, const float bls_alpha_lb, float * eta_rec, float * error_rec, float * bls_t_rec, const MKL_INT update_weights_n, const float in_ou_exc_tau, const float in_ou_exc_sigma)
{
    
    static float h_2i[NI]                    __attribute__((aligned(ALIGNMENT))) = {0},
                 h_2e[NE]                    __attribute__((aligned(ALIGNMENT))) = {0},
                 in_exc_current[NE]          __attribute__((aligned(ALIGNMENT))) = {0},
                 in_inh_current[NI]          __attribute__((aligned(ALIGNMENT))) = {0},
                 r_exc[NE]                   __attribute__((aligned(ALIGNMENT))) = {0},
                 r_inh[NI]                   __attribute__((aligned(ALIGNMENT))) = {0},
                 error[NE]                   __attribute__((aligned(ALIGNMENT))) = {0},
                 e2i_data_pos[E2I_NNZ]       __attribute__((aligned(ALIGNMENT))) = {0},
                 r_exc_rec_tmp[NE]           __attribute__((aligned(ALIGNMENT))) = {0},
                 r_inh_rec_tmp[NI]           __attribute__((aligned(ALIGNMENT))) = {0},
                 h_exc_rec_tmp[NE]           __attribute__((aligned(ALIGNMENT))) = {0},
                 h_inh_rec_tmp[NI]           __attribute__((aligned(ALIGNMENT))) = {0},
                 plastic_rec_tmp[N_PLASTIC]  __attribute__((aligned(ALIGNMENT))) = {0};
    float *plastic_data = e2i_data;
    const float minus_rho0 = -rho0, dt_tau_exc = dt / tau_exc, dt_tau_inh = dt / tau_inh, dt_tau_slow = dt / tau_slow;
    const MKL_INT plastic_array_size = N_PLASTIC,
                  n_e = NE, n_i = NI;
    MKL_INT curr_pattern_epoch = 0, curr_pattern_index = input_pattern_index[0], no_report = 0;
    const MKL_INT compute_on = steps_per_frame - 1, 
                  total_frames = n_steps / steps_per_frame, 
                  frames_per_report = n_steps / steps_per_frame / 100;
    int frame_n = 0;
    const float avg_div = 1.f/(float)steps_per_frame;
    mkl_set_num_threads_local(1);
    printf("Set number MKL threads %d\n", 1);

    
    
    plastic_nonlinearity(j_max, alpha_p, E2I_NNZ, e2i_data, e2i_data_pos);
    static float plastic_delta[N_PLASTIC] __attribute__((aligned(ALIGNMENT))) = {0},
                 i2e_delta[I2E_NNZ] __attribute__((aligned(ALIGNMENT))) = {0},
                 i2e_pre_rates[I2E_NNZ] __attribute__((aligned(ALIGNMENT))) = {0};
    const float *inh_thresh = NULL;
    const float *inh_gain_pos = NULL;
    printf("Using full exc2inh GRADIENT rule with max %f \n", (double)j_max);
    printf("INHIBITORY SYNAPTIC plasticity is enabled, with max %f \n", (double)j_i2e_max); 
    float *plastic_data_pos = __builtin_assume_aligned(e2i_data_pos, ALIGNMENT);
    const MKL_INT *x2i_data_i = e2i_data_i, *x2i_data_j = e2i_data_j;
    const float *r_pre = r_exc;
    
    VSLStreamStatePtr ou_stream;
    vslNewStream( &ou_stream, VSL_BRNG_SFMT19937, seed);
    printf("Init SFMT19937 stream using seed value %d \n", seed);
    
    float in_ou_exc_sigma_tau = sqrt(2 * dt * (float)pow(in_ou_exc_sigma, 2) / in_ou_exc_tau),
          norm_exc[NE] __attribute__((aligned(ALIGNMENT))),
          in_ou_exc_dt_tau = dt / in_ou_exc_tau;
    printf("Using OU input for excitatory population mu: %f sigma: %f \n", (double)in_exc_mean[0], (double)in_ou_exc_sigma);
     
    
    static float h_slow_exc[NE] __attribute__((aligned(ALIGNMENT))),
                 slow_delta[NE] __attribute__((aligned(ALIGNMENT)));
    cblas_scopy(NE, h_exc, 1, h_slow_exc, 1);
    
    printf("Target rate is %g Hz\n", (double)rho0);

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

    printf("Started C loop \n");
    
    for(MKL_INT n = 0; n < n_steps; ++n)
    {
        curr_pattern_epoch += (MKL_INT)(input_pattern_epochs[curr_pattern_epoch] < n);
        curr_pattern_index = input_pattern_index[curr_pattern_epoch];
        
    
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, ou_stream, NE, norm_exc, 0, in_ou_exc_sigma_tau);
        cblas_saxpy(NE, -in_ou_exc_dt_tau, in_exc_state, 1, norm_exc, 1); 
        vsAdd(NE, norm_exc, in_exc_state, in_exc_state); 
     
        
        vsAdd(NE, &in_exc_patterns[NE*curr_pattern_index], in_exc_state, in_exc_current);
        vsAdd(NE, in_exc_mean, in_exc_current, in_exc_current);
        mkl_cspblas_scsrgemv(no_trans, &n_e, i2e_data, i2e_indptr, i2e_indices, r_inh, h_2e);
        vsSub(NE, in_exc_current, h_2e, in_exc_current);
        mkl_cspblas_scsrgemv(no_trans, &n_e, e2e_data, e2e_indptr, e2e_indices, r_exc, h_2e);
        vsAdd(NE, h_2e, in_exc_current, in_exc_current); 
        vsSub(NE, in_exc_current, h_exc, in_exc_current);        // Compute change to input, dh = h_exc(t+1) - h_exc(t)
        cblas_saxpy(NE, dt_tau_exc, in_exc_current, 1, h_exc, 1); // New input h_exc += dt*dh/tau
        rate_nonlinearity(f_max, alpha_r, NE, h_exc, r_exc, NULL, NULL);
        
        cblas_scopy(NI, in_inh_state, 1, in_inh_current, 1);
        vsAdd(NI, in_inh_mean, in_inh_current, in_inh_current);
        mkl_cspblas_scsrgemv(no_trans, &n_i, e2i_data_pos, e2i_indptr, e2i_indices, r_exc, h_2i);
        vsAdd(NI, h_2i, in_inh_current, in_inh_current);
        mkl_cspblas_scsrgemv(no_trans, &n_i, i2i_data, i2i_indptr, i2i_indices, r_inh, h_2i);
        vsSub(NI, in_inh_current, h_2i, in_inh_current); 
        vsSub(NI, in_inh_current, h_inh, in_inh_current);
        cblas_saxpy(NI, dt_tau_inh, in_inh_current, 1, h_inh, 1); // New input h_inh += dt*dh/tau
        rate_nonlinearity(f_max, alpha_r, NI, h_inh, r_inh, inh_thresh, inh_gain_pos);
        vsAdd(NE, r_exc, r_exc_rec_tmp, r_exc_rec_tmp); // Add to running average
        vsAdd(NI, r_inh, r_inh_rec_tmp, r_inh_rec_tmp); 
        vsAdd(NE, h_exc, h_exc_rec_tmp, h_exc_rec_tmp);
        vsAdd(NI, h_inh, h_inh_rec_tmp, h_inh_rec_tmp);
        vsAdd(N_PLASTIC, plastic_data_pos, plastic_rec_tmp, plastic_rec_tmp);
        
        
        vsSub(NE, h_exc, h_slow_exc, slow_delta);
        cblas_saxpy(NE, dt_tau_slow, slow_delta, 1, h_slow_exc, 1);
        
        if(n % update_weights_n == 0 && plasticity_on <= n && n < plasticity_off)
        {
            cblas_scopy(NE, h_slow_exc, 1, error, 1); // Compute the per-excitatory neuron error.
            cblas_saxpy(NE,  1.f, &minus_rho0, 0, error, 1);
            
            vsPackV(I2E_NNZ, error, i2e_data_i, i2e_delta); // Take post-synaptic error into weight change array.
            vsPackV(I2E_NNZ, r_inh, i2e_data_j, i2e_pre_rates); // Take pre-synaptic inh rates
            vsMul(I2E_NNZ, i2e_pre_rates, i2e_delta, i2e_delta);
            cblas_sscal(I2E_NNZ, eta_i2e, i2e_delta, 1); // Scale the weight change by eta.
            vsAdd(I2E_NNZ, i2e_delta, i2e_data, i2e_data);
            for(MKL_INT i = 0; i < I2E_NNZ; ++i) i2e_data[i] = i2e_data[i] < 0.f ? 0.f : i2e_data[i];
            for(MKL_INT i = 0; i < I2E_NNZ; ++i) i2e_data[i] = i2e_data[i] > j_i2e_max ? j_i2e_max : i2e_data[i];
            update_plastic(error, eta, j_max, plastic_data, plastic_delta, alpha_r, h_inh, h_slow_exc, in_exc_state, in_inh_state, h_exc, n_e, n_i, i2e_data, i2e_indptr, i2e_indices, e2i_data_pos, e2i_indptr, e2i_indices, bls_c, bls_tau, bls_alpha_lb, eta_rec, error_rec, bls_t_rec, e2e_data, e2e_indptr, e2e_indices, i2i_data, i2i_indptr, i2i_indices, plastic_data_pos, x2i_data_i, x2i_data_j, r_pre, alpha_p, inh_thresh, inh_gain_pos);
            
            printf("eta = np.loadtxt(StringIO('%.6a'), dtype=np.float32)\n", (double)eta);
            pe2("r_exc", r_exc, 1, NE);
            pe2("r_inh", r_inh, 1, NI);
            //pia("e2i_data_i", e2i_data_i, E2I_NNZ);
            //pia("e2i_data_j", e2i_data_j, E2I_NNZ);
            pe2("dw_", plastic_delta, 1, N_PLASTIC);
            psa("e2i", e2i_data, E2I_NNZ, NI, NE, e2i_indices, e2i_indptr);
            psa("i2e", i2e_data, I2E_NNZ, NE, NI, i2e_indices, i2e_indptr);
            pe2("e2i_data_pos", e2i_data_pos, 1, NI);
        }
        else if(n < n_calibrate && n % 10 == 0)
        {
            float r_mu_error = cblas_sasum(NE, r_exc, 1) / NE - rho_calibrate;
        }
    

        if((n % steps_per_frame) == compute_on)
        {
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
            
            memset(r_exc_rec_tmp, 0, sizeof(r_exc_rec_tmp));
            memset(r_inh_rec_tmp, 0, sizeof(r_inh_rec_tmp));
            memset(h_exc_rec_tmp, 0, sizeof(h_exc_rec_tmp));
            memset(h_inh_rec_tmp, 0, sizeof(h_inh_rec_tmp));
            memset(plastic_rec_tmp, 0, sizeof(plastic_rec_tmp));

            if(0)
            {
                const float one = 1.f;
                float r_exc_mu = cblas_sasum(NE, r_exc, 1) / NE;
                float h_exc_mu = cblas_sdot(NE, &one, 0, h_exc, 1) / NE; // cblas_sasum(NE, h_exc, 1) / NE;
                float r_inh_mu = cblas_sasum(NI, r_inh, 1) / NI;
                MKL_INT maxi_r = cblas_isamax(NE, r_exc, 1);

                printf("r_exc_mu %.2f (h_exc %.2f) RMSE %.1f, max %.1e, (r_inh %g)\n", 
                       (double)r_exc_mu, (double)h_exc_mu, (double)sqrt(sp_rates[frame_n]), (double)r_exc[maxi_r], (double)r_inh_mu);
                

                pe2("r_exc", r_exc, 1, NE);
                pe2("h_exc", h_exc, 1, NE);
            }
            frame_n++;
            if(++no_report >= frames_per_report)
            {
                no_report = 0;
                printf("Computing frame %d / %d (%3.3f)\n", frame_n, total_frames, (double)100.*frame_n/total_frames);
            }
            // printf("===================================================================== \n");
            // eta *= .99;
        }
        
        for(MKL_INT i = 0; i < I2E_NNZ; ++i)
            if(!isfinite(i2e_data[i]))
            {
                n = n_steps;
                break;
            }
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
            printf("STOPPING EARLY!\n");
    }
    
    printf("Finished C loop\n");

    vslSSDeleteTask( &task_avg_sqr_err );
    vslSSDeleteTask( &task_e2i );
    vslDeleteStream( &ou_stream );
    
    
}