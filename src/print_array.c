
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