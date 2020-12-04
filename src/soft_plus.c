
void non_linearity_thresh_gain(const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain)
{
    const float *r = __builtin_assume_aligned(x, ALIGNMENT);
    const float *_thresh = __builtin_assume_aligned(thresh, ALIGNMENT);
    float *w = __builtin_assume_aligned(y, ALIGNMENT);
    // y = gain * ln(1 + exp(alpha*(x - thresh))) / alpha

    if(thresh)
        vsSub(n, x, thresh, y);
    else
        cblas_scopy(n, x, 1, y, 1);
    cblas_sscal(n, alpha, y, 1);
    vsExp(n, y, y);
    vsLog1p(n, y, y);
    cblas_sscal(n, 1.f / alpha, y, 1);
    if(_thresh)
        for(int i = 0; i < n; ++i)
            w[i] = isfinite(w[i]) ? w[i] : r[i] - _thresh[i];
    else
        for(int i = 0; i < n; ++i)
            w[i] = isfinite(w[i]) ? w[i] : r[i];

    if(gain)
        vsMul(n, gain, w, w);
}

void derivative_nl_thresh_gain(const float alpha, const MKL_INT n,
                               const float *restrict x, float *restrict y,
                               const float *restrict thresh, const float *restrict gain, const derivative_wrt wrt)
{
    // Derivative of gain*S+(x - thresh), WRT to x or thresh.
    // If derivative was WRT to thresh, result must be negated!!

    // y = gain / (1 + exp(alpha*(thresh - x)))
    const float one = 1;

    if(thresh)
    {
        vsSub(n, thresh, x, y);
        cblas_sscal(n, alpha, y, 1);
    }
    else
    {
        cblas_scopy(n, x, 1, y, 1);
        cblas_sscal(n, -alpha, y, 1);
    }
    vsExp(n, y, y);
    cblas_saxpy(n, 1., &one, 0, y, 1);
    vsInv(n, y, y);
    if(gain)
        vsMul(n, gain, y, y);
    if(wrt == THRESH) // Multiply that by -1
        cblas_sscal(n, -1.f, y, 1);
}