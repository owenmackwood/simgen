
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