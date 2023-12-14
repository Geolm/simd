#ifndef __SIMD_MATH__H__
#define __SIMD_MATH__H__

#include "simd.h"

#define SIMD_MATH_TAU (6.28318530f)
#define SIMD_MATH_PI  (3.14159265f)
#define SIMD_MATH_PI2 (1.57079632f)

//----------------------------------------------------------------------------------------------------------------------
// from hlslpp
// measured precision with input [-PI; PI] : 0.000002
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
// measured precision with input [-1; 1] : 0.0001
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
// measured precision with input [-PI; PI] : 0.005
static inline simd_vector simd_approx_cos(simd_vector a)
{
    a = simd_mul(a, simd_splat(1.f / SIMD_MATH_TAU));
    a = simd_sub(a, simd_add(simd_splat(.25f), simd_floor(simd_add(a, simd_splat(.25f)))));
    a = simd_mul(a, simd_mul(simd_splat(16.f), simd_sub(simd_abs(a), simd_splat(.5f))));
    return simd_add(a, simd_mul(simd_splat(.225f), simd_mul(a, simd_sub(simd_abs(a), simd_splat(1.f)))));
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_approx_sin(simd_vector a)
{
    return simd_approx_cos(simd_sub(a, simd_splat(SIMD_MATH_PI2)));
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://stackoverflow.com/questions/3380628/fast-arc-cos-algorithm
// measured precision with input [-1; 1] : 0.02
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
// measured precision with input [-1; 1] : 0.00001
static inline simd_vector simd_approx_atan(simd_vector x)
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
// based on https://en.wikipedia.org/wiki/Alpha_max_plus_beta_min_algorithm
// measured relative precision : 0.0005
static inline simd_vector simd_vec2_approx_length(simd_vector x, simd_vector y)
{
    simd_vector abs_value_x = simd_abs(x);
    simd_vector abs_value_y = simd_abs(y);
    simd_vector min_value = simd_min(abs_value_x, abs_value_y);
    simd_vector max_value = simd_max(abs_value_x, abs_value_y);
    
    simd_vector approximation = simd_fmad(simd_splat(0.485968200201465f), min_value, simd_mul(simd_splat(0.898204193266868f), max_value));
    approximation = simd_max(max_value, approximation);
    
    // do one newton raphson iteration
    simd_vector sq_length = simd_fmad(x, x, simd_mul(y, y));
    return simd_mul(simd_add(approximation, simd_div(sq_length, approximation)), simd_splat(0.5f));
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

#endif
