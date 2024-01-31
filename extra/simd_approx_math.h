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

// max error : 6.520748138e-05, ~2.8x faster than simd_acos
static simd_vector simd_approx_acos(simd_vector x);

// max error : 6.520736497e-05, ~2.8x faster than simd_acos
static simd_vector simd_approx_asin(simd_vector x);

// max relative error : 1.727835275e-03, ~3.7x faster than simd_exp
static simd_vector simd_approx_exp(simd_vector x); 

// max error : 2.321995254e-07
static simd_vector simd_approx_exp2(simd_vector x);

static simd_vector simd_approx_log2(simd_vector x);

// max error : 6.066019088e-02
static simd_vector simd_approx_sqrt(simd_vector x);


//----------------------------------------------------------------------------------------------------------------------
// range reduction from hlslpp, polynomial computed with lolremez
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
// based on https://developer.download.nvidia.com/cg/asin.html
static inline simd_vector simd_approx_asin(simd_vector x)
{
    simd_vector negate = simd_select(simd_splat_zero(), simd_splat(1.f), simd_cmp_lt(x, simd_splat_zero()));
    x = simd_abs(x);
    simd_vector result = simd_polynomial4(x, (float[]){-0.0187293f, 0.0742610f, -0.2121144f, 1.5707288f});
    result = simd_sub(simd_splat(SIMD_MATH_PI2), simd_mul(simd_sqrt(simd_sub(simd_splat(1.f), x)), result));
    return simd_sub(result, simd_mul(simd_mul(simd_splat(2.f), result), negate));
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://github.com/redorav/hlslpp/blob/master/include/hlsl%2B%2B_vector_float8.h
static simd_vector simd_approx_exp2(simd_vector x)
{
    simd_vector invalid_mask = simd_isnan(x);
    simd_vector input_is_infinity = simd_cmp_eq(x, simd_splat_positive_infinity());
    simd_vector equal_to_zero = simd_cmp_eq(x, simd_splat_zero());
    simd_vector one = simd_splat(1.f);
    
    // clamp values
    x = simd_clamp(x, simd_splat(-127.f), simd_splat(127.f));

    simd_vector ipart = simd_floor(x);
    simd_vector fpart = simd_sub(x, ipart);

    simd_vectori i = simd_shift_left_i(simd_add_i(simd_convert_from_float(ipart), simd_splat_i(127)), 23);
    simd_vector expipart = simd_cast_from_int(i);

    // minimax polynomial fit of 2^x, in range [-0.5, 0.5[
    simd_vector expfpart = simd_polynomial6(fpart, (float[]) {1.8775767e-3f, 8.9893397e-3f, 5.5826318e-2f, 2.4015361e-1f, 6.9315308e-1f, 1.f});

    simd_vector result = simd_mul(expipart, expfpart);
    result = simd_select(result, one, equal_to_zero);
    result = simd_or(result, invalid_mask);
    result = simd_select(result, simd_splat_positive_infinity(), input_is_infinity); // +inf arg will be +inf
    return result;
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://github.com/redorav/hlslpp/blob/master/include/hlsl%2B%2B_vector_float8.h
static simd_vector simd_approx_log2(simd_vector x)
{
    simd_vector one = simd_splat(1.f);
    simd_vectori i = simd_cast_from_float(x);
    simd_vector e = simd_convert_from_int(simd_sub_i(simd_shift_right_i( simd_and_i(i, simd_splat_i(0x7F800000)), 23), simd_splat_i(127)));
    simd_vector m = simd_or(simd_cast_from_int(simd_and_i(i, simd_splat_i(0x007FFFFF))), one);

    // minimax polynomial fit of log2(x)/(x - 1), for x in range [1, 2[
    simd_vector p = simd_polynomial6(m, (float[]) {-3.4436006e-2f, 3.1821337e-1f, -1.2315303f, 2.5988452f, -3.3241990f, 3.1157899f});

    // this effectively increases the polynomial degree by one, but ensures that log2(1) == 0
    simd_vector result = simd_fmad(p, simd_sub(m, one), e);

    // we can't compute a logarithm beyond this value, so we'll mark it as -infinity to indicate close to 0
    result = simd_select(result, simd_splat_negative_infinity(), simd_cmp_le(result, simd_splat(-127.f)));

    // check for negative values and return NaN
    result = simd_select(result, simd_splat_nan(), simd_cmp_lt(x, simd_splat_zero()));

    return result;
}

//----------------------------------------------------------------------------------------------------------------------
// based on based on https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
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
static inline simd_vector simd_approx_sqrt(simd_vector x)
{
    simd_vectori i = simd_cast_from_float(x);
    i = simd_sub_i(i, simd_splat_i(1<<23));
    i = simd_shift_right_i(i, 1);
    i = simd_add_i(i, simd_splat_i(1<<29));
    return simd_cast_from_int(i);
}


#endif

