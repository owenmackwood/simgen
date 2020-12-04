
void piece_wise(const float max_y, const float alpha  __attribute__((unused)), const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain)
{
    assert(thresh == NULL && gain == NULL);
    x = __builtin_assume_aligned(x, ALIGNMENT);
    y = __builtin_assume_aligned(y, ALIGNMENT);

    for(int i = 0; i < n; ++i)
        y[i] = (x[i] > 0.f) ? x[i] : 0.f;
    for(int i = 0; i < n; ++i)
        y[i] = (y[i] <= max_y) ? y[i] : max_y;
}

void piece_wise_derivative(const float max_y, const float alpha  __attribute__((unused)), const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain, const derivative_wrt wrt)
{
    assert(thresh == NULL && gain == NULL && wrt == OTHER);
    x = __builtin_assume_aligned(x, ALIGNMENT);
    y = __builtin_assume_aligned(y, ALIGNMENT);
    const float epsilon = 1.f-9;

    for(int i = 0; i < n; ++i)
        y[i] = (x[i] > 0.f && x[i] <= max_y) ? 1.f : epsilon;
}