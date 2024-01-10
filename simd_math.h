#ifndef __SIMD_MATH__H__
#define __SIMD_MATH__H__

#include "simd.h"

//----------------------------------------------------------------------------------------------------------------------
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
// prototypes
static simd_vector simd_sin(simd_vector x); // max error : 0.000001
static simd_vector simd_cos(simd_vector a); // max error : 0.000001
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c);
static simd_vector simd_acos(simd_vector x); // // max error : 0.000068
simd_vector simd_atan2(simd_vector x, simd_vector y); // max error : 0.000002
simd_vector simd_log(simd_vector x); // no error, output equal to logf()
simd_vector simd_exp(simd_vector x); // no error, output equal to expf()


//----------------------------------------------------------------------------------------------------------------------
// from hlslpp
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
static inline simd_vector simd_cos(simd_vector a)
{
    return simd_sin(simd_sub(simd_splat(SIMD_MATH_PI2), a));
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://developer.download.nvidia.com/cg/acos.html
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



#endif
