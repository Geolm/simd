#ifndef __SIMD_MATH__H__
#define __SIMD_MATH__H__

#include "simd.h"

#define SIMD_MATH_TAU (6.28318530f)
#define SIMD_MATH_PI  (3.14159265f)
#define SIMD_MATH_PI2 (1.57079632f)

//----------------------------------------------------------------------------------------------------------------------
// from hlslpp
// max error with input [-PI; PI] : 0.000001
static inline simd_vector simd_sin(simd_vector x)
{
    simd_vector invtau = simd_splat(1.f/SIMD_MATH_TAU);
    simd_vector tau = simd_splat(SIMD_MATH_TAU);
    simd_vector pi2 = simd_splat(SIMD_MATH_PI2);

    // Range reduction (into [-pi, pi] range)
    // Formula is x = x - round(x / 2pi) * 2pi
    x = simd_sub(x, simd_mul(simd_round(simd_mul(x, invtau)), tau));

    simd_vector gt_pi2 = simd_cmp_gt(x, pi2);
    simd_vector lt_minus_pi2 = simd_cmp_lt(x, simd_neg(pi2));
    simd_vector ox = x;

    // Use identities/mirroring to remap into the range of the minimax polynomial
    simd_vector pi = simd_splat(SIMD_MATH_PI);
    x = simd_select(x, simd_sub(pi, ox), gt_pi2);
    x = simd_select(x, simd_sub(simd_neg(pi), ox), lt_minus_pi2);

    simd_vector x_squared = simd_mul(x, x);

    simd_vector c1 = simd_splat(1.f);
    simd_vector c3 = simd_splat(-1.6665578e-1f);
    simd_vector c5 = simd_splat(8.3109378e-3f);
    simd_vector c7 = simd_splat(-1.84477486e-4f);

    simd_vector result;
    result = simd_fmad(x_squared, c7, c5);
    result = simd_fmad(x_squared, result, c3);
    result = simd_fmad(x_squared, result, c1);
    result = simd_mul(result, x);

    return result;
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://developer.download.nvidia.com/cg/acos.html
// max error with input [-1; 1] : 0.000068
static inline simd_vector simd_acos(simd_vector x)
{
    simd_vector negate = simd_select(simd_splat_zero(), simd_splat(1.f), simd_cmp_lt(x, simd_splat_zero()));
    x = simd_abs(x);
    simd_vector result = simd_splat(-0.0187293f);
    result = simd_fmad(result, x, simd_splat(0.0742610f));
    result = simd_fmad(result, x, simd_splat(-0.2121144f));
    result = simd_fmad(result, x, simd_splat(1.5707288f));
    result = simd_mul(result, simd_sqrt(simd_sub(simd_splat(1.f), x)));
    result = simd_sub(result, simd_mul(simd_mul(simd_splat(2.f), negate), result));
    return simd_fmad(negate, simd_splat(SIMD_MATH_PI), result);
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_cos(simd_vector a)
{
    return simd_sin(simd_sub(simd_splat(SIMD_MATH_PI2), a));
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://stackoverflow.com/a/66868438
// max error with input [-PI; PI] : 0.057
static inline simd_vector simd_approx_cos(simd_vector a)
{
    a = simd_mul(a, simd_splat(1.f / SIMD_MATH_TAU));
    a = simd_sub(a, simd_add(simd_splat(.25f), simd_floor(simd_add(a, simd_splat(.25f)))));
    return simd_mul(a, simd_mul(simd_splat(16.f), simd_sub(simd_abs(a), simd_splat(.5f))));
    // note : we don't use extra precision because it slow down too much the function to be worth it compared to simd_sin
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_approx_sin(simd_vector a)
{
    return simd_approx_cos(simd_sub(a, simd_splat(SIMD_MATH_PI2)));
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://stackoverflow.com/questions/3380628/fast-arc-cos-algorithm
// max error with input [-1; 1] : 0.016723
static inline simd_vector simd_approx_acos(simd_vector x)
{
    simd_vector x_2 = simd_mul(x, x);
    simd_vector x_3 = simd_mul(x_2, x);
    simd_vector x_4 = simd_mul(x_2, x_2);
    simd_vector result = simd_mul(simd_splat(-0.939115566365855f), x);
    result = simd_fmad(simd_splat(0.9217841528914573f), x_3, result);
    
    simd_vector divisor = simd_sub(simd_splat(1.f), simd_mul(simd_splat(1.2845906244690837f), x_2));
    divisor = simd_fmad(simd_splat(0.295624144969963174f), x_4, divisor);
    result = simd_div(result, divisor);

    return simd_add(result, simd_splat(SIMD_MATH_PI2));
}

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
static inline simd_vector simd_atan2(simd_vector x, simd_vector y)
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

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_smoothstep(simd_vector edge0, simd_vector edge1, simd_vector x)
{
    x = simd_saturate(simd_div(simd_sub(x, edge0), simd_sub(edge1, edge0)));
    return simd_mul(simd_mul(x, x), simd_sub(simd_splat(3.f), simd_add(x, x)));
}

//-----------------------------------------------------------------------------
static inline simd_vector simd_quadratic_bezier(simd_vector p0, simd_vector p1, simd_vector p2, simd_vector t)
{
    simd_vector one_minus_t = simd_sub(simd_splat(1.f), t);
    simd_vector a = simd_mul(one_minus_t, one_minus_t);
    simd_vector b = simd_mul(simd_splat(2.f), simd_mul(one_minus_t, t));
    simd_vector c = simd_mul(t, t);
    return simd_fmad(p0, a, simd_fmad(p1, b, simd_mul(p2, c)));
}

//-----------------------------------------------------------------------------
// based on https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
// max relative error with input [-87.33654; 88.72283] : 0.001726
static inline simd_vector simd_approx_exp(simd_vector x)
{
    simd_vector c0 = simd_splat(0.3371894346f);
    simd_vector c1 = simd_splat(0.657636276f);
    simd_vector c2 = simd_splat(1.00172476f);

    // exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1
    simd_vector t = simd_mul(x, simd_splat(1.442695041f)); // t = log2(e) * x
    simd_vector e = simd_floor(t);

    simd_vector f = simd_sub(t, e); // f = t - floor(t)
    simd_vector p = simd_fmad(c0, f, c1); // p = c0 * f + c1
    p = simd_fmad(p, f, c2); // p = (c0 * f + c1) * f + c2

#if defined(SIMD_NEON_IMPLEMENTATION)
    int32x4_t i = vcvtnq_s32_f32(e);
    i = vshlq_s32(i, vdupq_n_s32(23));
    return vreinterpretq_s32_f32(vaddq_s32(i, vreinterpretq_f32_s32(p)));
#elif defined(SIMD_AVX_IMPLEMENTATION)
    __m256i i = _mm256_cvtps_epi32(e);
    i = _mm256_slli_epi32(i, 23);
    return _mm256_castsi256_ps(_mm256_add_epi32(i, _mm256_castps_si256(p)));
#endif
}

//-----------------------------------------------------------------------------
static inline simd_vector simd_log(simd_vector x)
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
// based on infinite binomial serie
// works only for input in [0; 2]
static inline simd_vector simd_approx_pow(simd_vector x, simd_vector exponent, uint32_t degree)
{
    simd_vector one = simd_splat(1.f);
    simd_vector a = exponent;
    simd_vector k = one;
    simd_vector x_minus_one = simd_sub(x, one);
    simd_vector divisor = k;
    simd_vector factor = x_minus_one;
    simd_vector result = simd_fmad(simd_div(a, divisor), factor, one);

    for(uint32_t i=1; i<=degree; ++i)
    {
        k = simd_add(k, one);
        divisor = simd_mul(divisor, k);
        a = simd_mul(a, simd_add(simd_sub(exponent, k), one));
        factor = simd_mul(factor, x_minus_one);
        result = simd_fmad(simd_div(a, divisor), factor, result);
    }
    return result;
}

//-----------------------------------------------------------------------------
// max error : 0.015
static inline simd_vector simd_approx_linear_to_srgb(simd_vector value)
{
    simd_vector big_value = simd_cmp_ge(value, simd_splat(0.0031308f));
    simd_vector result0 = simd_mul(value, simd_splat(12.92f));

    // pow(x, 1/2.4) approximation : two polymoms combined

    //0.066832 + 9.159853*x^1 + -161.614105*x^2 + 1343.757080*x^3 + 122.443939*x^4 + -1312.813599*x^5
    //average_error: 0.001049 max_error:0.003493 in range [0.003131; 0.050000]
    // 0.147392 + 2.841041*x^1 + -6.972846*x^2 + 11.236314*x^3 + -9.159635*x^4 + 2.907734*x^5
    // average_error: 0.003165 max_error:0.065744 in range [0.003131; 1.000000]
    
    simd_vector above_0_05 = simd_cmp_gt(value, simd_splat(0.05f));
    simd_vector result1 = simd_select(simd_splat(-1312.813599f), simd_splat(2.907734f), above_0_05);
    result1 = simd_fmad(result1, value, simd_select(simd_splat(122.443939f), simd_splat(-9.159635f), above_0_05));
    result1 = simd_fmad(result1, value, simd_select(simd_splat(1343.757080f), simd_splat(11.236314f), above_0_05));
    result1 = simd_fmad(result1, value, simd_select(simd_splat(-161.614105f), simd_splat(-6.972846f), above_0_05));
    result1 = simd_fmad(result1, value, simd_select(simd_splat(9.159853f), simd_splat(2.841041f), above_0_05));
    result1 = simd_fmad(result1, value, simd_select(simd_splat(0.066832f), simd_splat(0.147392f), above_0_05));

    result1 = simd_fmad(result1, simd_splat(1.055f), simd_splat(-0.055f));
    return simd_select(result0, result1, big_value);
}

//-----------------------------------------------------------------------------
// max error : 0.000079
static inline simd_vector simd_approx_srgb_to_linear(simd_vector value)
{
    simd_vector big_value = simd_cmp_ge(value, simd_splat(0.04045f));
    simd_vector result0 = simd_mul(value, simd_splat(1.f / 12.92f));
    value = simd_mul(simd_add(value, simd_splat(0.055f)), simd_splat(1.f / 1.055f));

    // pow(x, 2.4) approximation
    // 0.000589 + -0.021682*x + 0.471282*x^2 + 0.801800*x^3 + -0.335099*x^4 + 0.083109*x^5
    // average_error: 0.000023 max_error:0.000082 in range [0.040450; 1.000000]
    simd_vector result1 = simd_splat(0.083109f);
    result1 = simd_fmad(result1, value, simd_splat(-0.335099f));
    result1 = simd_fmad(result1, value, simd_splat(0.801800f));
    result1 = simd_fmad(result1, value, simd_splat(0.471282f));
    result1 = simd_fmad(result1, value, simd_splat(-0.021682f));
    result1 = simd_fmad(result1, value, simd_splat(0.000589f));
    return simd_select(result0, result1, big_value);
}

// TODO:
// hsv to rgb based on https://github.com/stolk/hsvbench/blob/main/hsv.h
// simd_log and simd_exp based on http://gruntthepeon.free.fr/ssemath/sse_mathfun.h

#endif
