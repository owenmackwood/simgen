
void clip_upper(const MKL_INT n, float *v, const float max)
{
    v = __builtin_assume_aligned(v, ALIGNMENT);
    for(MKL_INT i = 0; i < n; ++i)
        v[i] = (max < v[i]) ? max : v[i];
}

void clip_lower(const MKL_INT n, float *v, const float min)
{
    v = __builtin_assume_aligned(v, ALIGNMENT);
    for(MKL_INT i = 0; i < n; ++i)
        v[i] = (v[i] < min) ? min : v[i];
}

void clip(const MKL_INT n, float *v, const float min, const float max)
{
    clip_upper(n, v, max);
    clip_lower(n, v, min);
    // _v = _mm256_blend_ps(_v, _min, _mm256_cmp_ps(_v, _min, _CMP_LT_OQ)); // AVX res = a < b, Ordered, Quiet
    // _v = _mm256_blend_ps(_max, _v, _mm256_cmp_ps(_max, _v, _CMP_LT_OQ));
}

/*
    float *w = __builtin_assume_aligned(v, ALIGNMENT);
    __m256 _v;
    const int floats_per_vec = 8;
    const __m256 _max = _mm256_set_ps(max, max, max, max, max, max, max, max);
    const int m = n / floats_per_vec;
    for(MKL_INT i = 0; i < m; ++i)
    {{
        _v = _mm256_load_ps(w + i*floats_per_vec);
        _v = _mm256_min_ps(_v, _max);
        _mm256_store_ps(w + i*floats_per_vec, _v);
    }}

    for(MKL_INT i = m * floats_per_vec; i < n; ++i)
        w[i] = (max < w[i]) ? max : w[i];
    */
    /*
    float *w = __builtin_assume_aligned(v, ALIGNMENT);
    __m256 _v;
    const int floats_per_vec = 8;
    const __m256 _min = _mm256_set_ps(min, min, min, min, min, min, min, min);
    const int m = n / floats_per_vec;
    for(MKL_INT i = 0; i < m; ++i)
    {{
        _v = _mm256_load_ps(w + i*floats_per_vec);
        _v = _mm256_max_ps(_v, _min);
        _mm256_store_ps(w + i*floats_per_vec, _v);
    }}

    for(MKL_INT i = m * floats_per_vec; i < n; ++i)
        w[i] = (w[i] < min) ? min : w[i];
*/
