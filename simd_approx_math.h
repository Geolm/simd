#ifndef __SIMD_APPROX_MATH__H__
#define __SIMD_APPROX_MATH__H__

#include "simd.h"

#ifndef SIMD_MATH_CONSTANTS
    #define SIMD_MATH_CONSTANTS
    #define SIMD_MATH_TAU (6.28318530f)
    #define SIMD_MATH_PI  (3.14159265f)
    #define SIMD_MATH_PI2 (1.57079632f)
    #define SIMD_MATH_PI4 (0.78539816f)
#endif

//----------------------------------------------------------------------------------------------------------------------
// Common math functions approximation
//
// Compromise between speed and precision
// You can run the benchmark to see time difference
//
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
// Prototypes
static simd_vector simd_approx_cos(simd_vector a); // max error : 0.000001, ~2.5x faster than simd_cos
static simd_vector simd_approx_sin(simd_vector a); // max error : 0.000001, ~2.5x faster than simd_cos
static simd_vector simd_approx_acos(simd_vector x); // max error : 0.000068, ~2.8x faster than simd_acos
static simd_vector simd_approx_exp(simd_vector x); // max relative error : 0.001726, ~3.7x faster than simd_cos
static simd_vector simd_approx_srgb_to_linear(simd_vector value); // max error : 0.000079
static simd_vector simd_approx_linear_to_srgb(simd_vector value); // max error : 0.003851 (enough for 8bits value)

//----------------------------------------------------------------------------------------------------------------------
// from hlslpp
static inline simd_vector simd_approx_sin(simd_vector x)
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
static inline simd_vector simd_approx_cos(simd_vector x)
{
    return simd_approx_sin(simd_sub(simd_splat(SIMD_MATH_PI2), x));
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://developer.download.nvidia.com/cg/acos.html
static inline simd_vector simd_approx_acos(simd_vector x)
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

    simd_vectori i = simd_convert_from_float(e);
    i = simd_shift_left_i(i, 23);
    return simd_cast_from_int(simd_add_i(i, simd_cast_from_float(p)));
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_approx_linear_to_srgb(simd_vector value)
{
    // range [0.000000; 0.042500]
    // f(x) = 0.003309 + 12.323890*x^1 + -307.151428*x^2 + 3512.375244*x^3 + -3457.260986*x^4 + 240.709824*x^5 
    
    // Degree 5 approximation of f(x) = 1.055 * pow(x, 1/2.4) - 0.055
    // on interval [ 0.042500, 1 ]
    // p(x)=((((3.7378898*x-1.1148316e+1)*x+1.2821273e+1)*x-7.416023)*x+2.887472)*x+1.2067896e-1
    // Estimated max error: 2.9748487e-3
    simd_vector above_threshold = simd_cmp_gt(value, simd_splat(.0425f));
    simd_vector result = simd_select(simd_splat(240.709824f), simd_splat(3.7378898f), above_threshold);
    result = simd_fmad(result, value, simd_select(simd_splat(-3457.260986f), simd_splat(-11.148315f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(3512.375244f), simd_splat(12.821273f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(-307.151428f), simd_splat(-7.4160228f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(12.323890f), simd_splat(2.8874719f), above_threshold));
    result = simd_fmad(result, value, simd_select(simd_splat(0.003309f), simd_splat(0.12067895f), above_threshold));
    return result;
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_approx_srgb_to_linear(simd_vector value)
{
    simd_vector big_value = simd_cmp_ge(value, simd_splat(0.04045f));
    simd_vector result0 = simd_mul(value, simd_splat(1.f / 12.92f));
    value = simd_mul(simd_add(value, simd_splat(0.055f)), simd_splat(1.f / 1.055f));
    
    // Degree 5 approximation of f(x) = pow(x, 2.4)
    // on interval [ 0.04045, 1 ]
    // p(x)=((((1.0174202e-1*x-3.8574015e-1)*x+8.5166867e-1)*x+4.5003571e-1)*x-1.8100243e-2)*x+4.3660368e-4
    // Estimated max error: 4.2615527e-5
    simd_vector result1 = simd_splat(0.10174202f);
    result1 = simd_fmad(result1, value, simd_splat(-0.38574016f));
    result1 = simd_fmad(result1, value, simd_splat(0.85166866f));
    result1 = simd_fmad(result1, value, simd_splat(0.45003572f));
    result1 = simd_fmad(result1, value, simd_splat(-0.018100243f));
    result1 = simd_fmad(result1, value, simd_splat(0.00043660367f));
    return simd_select(result0, result1, big_value);
}

// TODO:
// hsv to rgb based on https://github.com/stolk/hsvbench/blob/main/hsv.h


#endif

