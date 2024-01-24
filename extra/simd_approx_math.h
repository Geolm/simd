#ifndef __SIMD_APPROX_MATH__H__
#define __SIMD_APPROX_MATH__H__

#include "../simd.h"

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

// max error : 5.811452866e-07, ~2.5x faster than simd_cos
static simd_vector simd_approx_cos(simd_vector a); 

// max error : 2.682209015e-07, ~2.5x faster than simd_sin
static simd_vector simd_approx_sin(simd_vector a); 

// max error : 0.000068, ~2.8x faster than simd_acos
static simd_vector simd_approx_acos(simd_vector x);

// max relative error : 0.001726, ~3.7x faster than simd_exp
static simd_vector simd_approx_exp(simd_vector x); 


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
    simd_vector result = simd_polynomial4(x_squared, (float[]){2.6000548e-6f, -1.9806615e-4f, 8.3330173e-3f, -1.6666657e-1f});

    result = simd_mul(result, x_squared);
    result = simd_fmad(result, x, x);

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
    simd_vector result = simd_polynomial4(x, (float[]){-0.0187293f, 0.0742610f, -0.2121144f, 1.5707288f});
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


#endif

