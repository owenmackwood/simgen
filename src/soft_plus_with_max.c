
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