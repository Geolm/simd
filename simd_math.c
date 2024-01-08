#include "simd_math.h"

//-----------------------------------------------------------------------------
// based on http://gruntthepeon.free.fr/ssemath/
simd_vector simd_log(simd_vector x)
{
    simd_vector one = simd_splat(1.f);
    simd_vector invalid_mask = simd_cmp_le(x, simd_splat_zero());
    x = simd_max(x, simd_min_normalized());  // cut off denormalized stuff

#if defined(SIMD_NEON_IMPLEMENTATION)
    int32x4_t emm0 = vshlq_s32(vreinterpretq_f32_s32(x), vdupq_n_s32(-23));
    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    simd_vector e = vcvtq_f32_s32(emm0);
#elif defined(SIMD_AVX_IMPLEMENTATION)
    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
    simd_vector e = _mm256_cvtepi32_ps(emm0);
#endif
    
    // keep only the fractional part
    x = simd_and(x, simd_inv_mant_mask());
    x = simd_or(x, simd_splat(0.5f));
    
    e = simd_add(e, one);
    simd_vector mask = simd_cmp_lt(x, simd_splat(0.707106781186547524f));
    simd_vector tmp = simd_and(x, mask);
    x = simd_sub(x, one);
    e = simd_sub(e, simd_and(one, mask));
    x = simd_add(x, tmp);

    simd_vector z = simd_mul(x,x);
    simd_vector y = simd_splat(7.0376836292E-2f);
    y = simd_fmad(y, x, simd_splat(-1.1514610310E-1f));
    y = simd_fmad(y, x, simd_splat(1.1676998740E-1f));
    y = simd_fmad(y, x, simd_splat(-1.2420140846E-1f));
    y = simd_fmad(y, x, simd_splat(+1.4249322787E-1f));
    y = simd_fmad(y, x, simd_splat(-1.6668057665E-1f));
    y = simd_fmad(y, x, simd_splat(+2.0000714765E-1f));
    y = simd_fmad(y, x, simd_splat(-2.4999993993E-1f));
    y = simd_fmad(y, x, simd_splat(+3.3333331174E-1f));
    y = simd_mul(y, x);
    y = simd_mul(y, z);

    tmp = simd_mul(e, simd_splat(-2.12194440e-4f));
    y = simd_add(y, tmp);

    tmp = simd_mul(z, simd_splat(0.5f));
    y = simd_sub(y, tmp);

    tmp = simd_mul(e, simd_splat(0.693359375f));
    x = simd_add(x, y);
    x = simd_add(x, tmp);
    x = simd_or(x, invalid_mask); // negative arg will be NAN

    return x;
}

//-----------------------------------------------------------------------------
// based on http://gruntthepeon.free.fr/ssemath/
simd_vector simd_exp(simd_vector x)
{
    simd_vector tmp = simd_splat_zero();
    simd_vector fx;
    simd_vector one = simd_splat(1.f);

    x = simd_min(x, simd_splat(88.3762626647949f));
    x = simd_max(x, simd_splat(-88.3762626647949f));

    // express exp(x) as exp(g + n*log(2))
    fx = simd_fmad(x, simd_splat(1.44269504088896341f), simd_splat(0.5f));
    tmp = simd_floor(fx);

    // if greater, substract 1
    simd_vector mask = simd_cmp_gt(tmp, fx);
    mask = simd_and(mask, one);
    fx = simd_sub(tmp, mask);

    tmp = simd_mul(fx, simd_splat(0.693359375f));
    simd_vector z = simd_mul(fx, simd_splat(-2.12194440e-4f));
    x = simd_sub(x, tmp);
    x = simd_sub(x, z);
    z = simd_mul(x, x);
    simd_vector y = simd_splat(1.9875691500E-4f);
    y = simd_fmad(y, x, simd_splat(1.3981999507E-3f));
    y = simd_fmad(y, x, simd_splat(8.3334519073E-3f));
    y = simd_fmad(y, x, simd_splat(4.1665795894E-2f));
    y = simd_fmad(y, x, simd_splat(1.6666665459E-1f));
    y = simd_fmad(y, x, simd_splat(5.0000001201E-1f));
    y = simd_fmad(y, z, x);
    y = simd_add(y, one);

#if defined(SIMD_NEON_IMPLEMENTATION)
    int32x4_t emm0 = vcvtq_s32_f32(fx);
    emm0 = vaddq_s32(emm0, vdupq_n_s32(0x7f));
    emm0 = vshlq_s32(emm0, vdupq_n_s32(23));
    simd_vector pow2n = vreinterpretq_s32_f32(emm0);
#elif defined(SIMD_AVX_IMPLEMENTATION)
    __m256i emm0 = _mm256_cvtps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    simd_vector pow2n = _mm256_castsi256_ps(emm0);
#endif

    y = simd_mul(y, pow2n);
    return y;
}
