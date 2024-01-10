#ifndef __SIMD_APPROX_MATH__H__
#define __SIMD_APPROX_MATH__H__

#include "simd.h"

#ifndef SIMD_MATH_TAU
    #define SIMD_MATH_TAU (6.28318530f)
#endif

#ifndef SIMD_MATH_PI
    #define SIMD_MATH_PI  (3.14159265f)
#endif

#ifndef SIMD_MATH_PI2
    #define SIMD_MATH_PI2 (1.57079632f)
#endif

//----------------------------------------------------------------------------------------------------------------------
// Common math functions approximation : speed over precision
//
// Use this function if you *really* don't need a lot of precision and you need to be fast
// In most cases, you should use functions from simd_math.h (run the benchmark to see time difference)
//
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
// Prototypes
static simd_vector simd_approx_cos(simd_vector a); // max error : 0.057
static simd_vector simd_approx_sin(simd_vector a); // max error : 0.057
static simd_vector simd_approx_acos(simd_vector x); // max error : 0.016723
static simd_vector simd_approx_exp(simd_vector x); // max relative error : 0.001726
static simd_vector simd_approx_srgb_to_linear(simd_vector value); // max error : 0.000079
static simd_vector simd_approx_linear_to_srgb(simd_vector value); // max error : 0.003851

//----------------------------------------------------------------------------------------------------------------------
// based on https://stackoverflow.com/a/66868438
// note : we don't use extra precision code because it slows down too much the function to be worth it compared to simd_sin
static inline simd_vector simd_approx_cos(simd_vector a)
{
    a = simd_mul(a, simd_splat(1.f / SIMD_MATH_TAU));
    a = simd_sub(a, simd_add(simd_splat(.25f), simd_floor(simd_add(a, simd_splat(.25f)))));
    return simd_mul(a, simd_mul(simd_splat(16.f), simd_sub(simd_abs(a), simd_splat(.5f))));
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_approx_sin(simd_vector a)
{
    return simd_approx_cos(simd_sub(a, simd_splat(SIMD_MATH_PI2)));
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://stackoverflow.com/questions/3380628/fast-arc-cos-algorithm
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
// based on https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
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
static inline simd_vector simd_approx_linear_to_srgb(simd_vector value)
{
    // range [0.000000; 0.042500]
    // f(x) = 0.003309 + 12.323890*x^1 + -307.151428*x^2 + 3512.375244*x^3 + -3457.260986*x^4 + 240.709824*x^5 
    // range [0.042500: 1.000000]
    // f(x) = 0.125607 + 2.752478*x^1 + -6.438252*x^2 + 10.166183*x^3 + -8.150837*x^4 + 2.544821*x^5 
    // max error: 0.003827
    simd_vector above_threshold = simd_cmp_gt(value, simd_splat(.0425f));
    simd_vector result = simd_select(simd_splat(240.709824f), simd_splat(2.544821f), above_threshold);
    result = simd_fmad(result, value, simd_select(simd_splat(-3457.260986f), simd_splat(-8.150837f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(3512.375244f), simd_splat(10.166183f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(-307.151428f), simd_splat(-6.438252f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(12.323890f), simd_splat(2.752478f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(0.003309f), simd_splat(0.125607f), above_threshold));
    return result;
}

//-----------------------------------------------------------------------------
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


#endif

