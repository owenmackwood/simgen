
void non_linearity_thresh_gain(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain)
{
    // y = max_y / (1 + exp(alpha * (1/2 - x/max_y)))

    assert(thresh == NULL && gain == NULL);
    x = __builtin_assume_aligned(x, ALIGNMENT);
    y = __builtin_assume_aligned(y, ALIGNMENT);
    for(int i = 0; i < n; ++i)
        y[i] = alpha * (.5f - x[i] / max_y);
    vsExp(n, y, y);
    for(int i = 0; i < n; ++i)
        y[i] = max_y / (1.f + y[i]);
    /*
    cblas_scopy(n, x, 1, y, 1);
    cblas_sscal(n, -1.f / max_y, y, 1);
    cblas_saxpy(n,  1.f, &one_half, 0, y, 1);
    cblas_sscal(n, alpha, y, 1);
    vsExp(n, y, y);
    cblas_saxpy(n,  1.f, &one, 0, y, 1);
    vsInv(n, y, y);
    cblas_sscal(n, max_y, y, 1);
    */
    for(int i = 0; i < n; ++i)
        assert(isfinite(y[i]));
}

void derivative_nl_thresh_gain(const float max_y, const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain, const derivative_wrt wrt)
{
    // yp = alpha/4* cosh(alpha*(max_y-2*x)/(4*max_y))**-2

    assert(wrt == OTHER && thresh == NULL && gain == NULL);
    x = __builtin_assume_aligned(x, ALIGNMENT);
    y = __builtin_assume_aligned(y, ALIGNMENT);

    for(int i = 0; i < n; ++i)
        y[i] = alpha * (max_y - 2*x[i]) / (4*max_y);
    vsCosh(n, y, y);
    vsPowx(n, y, -2.f, y);
    for(int i = 0; i < n; ++i)
        y[i] *= alpha/4.f;
}