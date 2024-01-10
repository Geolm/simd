#include "simd_math.h"

//-----------------------------------------------------------------------------
// https://mazzo.li/posts/vectorized-atan2.html
// max error with input [-1; 1] : 0.000002
// input SHOULD be in [-1; 1]
static inline simd_vector simd_atan(simd_vector x)
{
    simd_vector a1  = simd_splat(0.99997726f);
    simd_vector a3  = simd_splat(-0.33262347f);
    simd_vector a5  = simd_splat(0.19354346f);
    simd_vector a7  = simd_splat(-0.11643287f);
    simd_vector a9  = simd_splat(0.05265332f);
    simd_vector a11 = simd_splat(-0.01172120f);
    simd_vector x_sq = simd_mul(x, x);

    return simd_mul(x, simd_fmad(x_sq, simd_fmad(x_sq, simd_fmad(x_sq, simd_fmad(x_sq, simd_fmad(x_sq, a11, a9), a7), a5), a3), a1));
}

//-----------------------------------------------------------------------------
// https://mazzo.li/posts/vectorized-atan2.html
// max error : 0.000002
simd_vector simd_atan2(simd_vector x, simd_vector y)
{
    simd_vector swap = simd_cmp_lt(simd_abs(x), simd_abs(y));
    simd_vector x_over_y = simd_div(x, y);
    simd_vector y_over_x = simd_div(y, x);
    simd_vector atan_input = simd_select(y_over_x, x_over_y, swap);
    simd_vector result = simd_atan(atan_input);

    simd_vector adjust = simd_select(simd_splat(-SIMD_MATH_PI2), simd_splat(SIMD_MATH_PI2), simd_cmp_ge(atan_input, simd_splat_zero()));
    result = simd_select(result, simd_sub(adjust, result), swap);

    simd_vector x_sign_mask = simd_cmp_lt(x, simd_splat_zero());
    return simd_add( simd_and(simd_xor(simd_splat(SIMD_MATH_PI), simd_and(simd_sign_mask(), y)), x_sign_mask), result);
}

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

//-----------------------------------------------------------------------------
// based on http://gruntthepeon.free.fr/ssemath/
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c)
{
    simd_vector xmm1, xmm2, xmm3 = simd_splat_zero(), sign_bit_sin, y;

    sign_bit_sin = x;

    // take the absolute value
    x = simd_and(x, simd_inv_sign_mask());
    // extract the sign bit (upper one)
    sign_bit_sin = simd_and(sign_bit_sin, simd_sign_mask());

    // scale by 4/Pi
    y = simd_mul(x, simd_splat(1.27323954473516f));

#if defined(SIMD_NEON_IMPLEMENTATION)
    int32x4_t emm2 = vcvtq_s32_f32(y);

    emm2 = vaddq_s32(emm2, vdupq_n_u32(1));
    emm2 = vandq_s32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_s32(emm2);

    int32x4_t emm4 = emm2;

    // get the swap sign flag for the sine
    int32x4_t emm0 = vandq_s32(emm2, vdupq_n_u32(4));
    emm0 = vshlq_s32(emm0, vdupq_n_s32(29));
    simd_vector swap_sign_bit_sin = vreinterpretq_s32_f32(emm0);

    // get the polynom selection mask for the sine
    emm2 = vandq_s32(emm2, vdupq_n_u32(2));
    emm2 = vcgeq_s32(emm2, vdupq_n_u32(0));
    simd_vector poly_mask = vreinterpretq_s32_f32(emm2);

#elif defined(SIMD_AVX_IMPLEMENTATION)
    // store the integer part of y in emm2 
    __m256i emm2 = _mm256_cvttps_epi32(y);

    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(emm2);

    __m256i emm4 = emm2;

    // get the swap sign flag for the sine
    __m256i emm0 = _mm256_and_si256(emm2, _mm256_set1_epi32(4));
    emm0 = _mm256_slli_epi32(emm0, 29);
    simd_vector swap_sign_bit_sin = _mm256_castsi256_ps(emm0);

    // get the polynom selection mask for the sine
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(2));
    emm2 = _mm256_cmpeq_epi32(emm2, _mm256_castps_si256(_mm256_setzero_ps()));
    simd_vector poly_mask = _mm256_castsi256_ps(emm2);
#endif

    // The magic pass: "Extended precision modular arithmetic" 
    //  x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = simd_fmad(y, simd_splat(-0.78515625f), x);
    x = simd_fmad(y, simd_splat(-2.4187564849853515625e-4f), x);
    x = simd_fmad(y, simd_splat(-3.77489497744594108e-8f), x);

#if defined(SIMD_NEON_IMPLEMENTATION)
    emm4 = vsubq_s32(emm4, vdupq_n_u32(2));
    emm4 = vbicq_s32(emm4, vdupq_n_u32(4));
    emm4 = vshlq_s32(emm4, vdupq_n_s32(29));
    simd_vector sign_bit_cos = vreinterpretq_s32_f32(emm4);
#elif defined(SIMD_AVX_IMPLEMENTATION)
    emm4 = _mm256_sub_epi32(emm4, _mm256_set1_epi32(2));
    emm4 = _mm256_andnot_si256(emm4, _mm256_set1_epi32(4));
    emm4 = _mm256_slli_epi32(emm4, 29);
    simd_vector sign_bit_cos = _mm256_castsi256_ps(emm4);
#endif

    sign_bit_sin = simd_xor(sign_bit_sin, swap_sign_bit_sin);
    
    // Evaluate the first polynom  (0 <= x <= Pi/4)
    simd_vector z = simd_mul(x,x);
    y = simd_splat(2.443315711809948E-005f);
    y = simd_fmad(y, z, simd_splat(-1.388731625493765E-003f));
    y = simd_fmad(y, z, simd_splat(4.166664568298827E-002f));
    y = simd_mul(y, z);
    y = simd_mul(y, z);
    simd_vector tmp = simd_mul(z, simd_splat(.5f));
    y = simd_sub(y, tmp);
    y = simd_add(y, simd_splat(1.f));

    // Evaluate the second polynom  (Pi/4 <= x <= 0)
    simd_vector y2 = simd_splat(-1.9515295891E-4f);
    y2 = simd_fmad(y2, z, simd_splat(8.3321608736E-3f));
    y2 = simd_fmad(y2, z, simd_splat(-1.6666654611E-1f));
    y2 = simd_mul(y2, z);
    y2 = simd_mul(y2, x);
    y2 = simd_add(y2, x);

    // select the correct result from the two polynoms
    xmm3 = poly_mask;
    simd_vector ysin2 = simd_and(xmm3, y2);
    simd_vector ysin1 = simd_andnot(xmm3, y);
    y2 = simd_sub(y2,ysin2);
    y = simd_sub(y, ysin1);

    xmm1 = simd_add(ysin1,ysin2);
    xmm2 = simd_add(y,y2);

    // update the sign
    *s = simd_xor(xmm1, sign_bit_sin);
    *c = simd_xor(xmm2, sign_bit_cos);
}



