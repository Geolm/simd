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
static simd_vector simd_sin(simd_vector x); // output equal to sinf()
static simd_vector simd_cos(simd_vector a); // output equal to cosf()
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c); // output equal to sinf()/cosf()
static simd_vector simd_acos(simd_vector x); // // max error : 0.000068
simd_vector simd_atan2(simd_vector x, simd_vector y); // max error : 0.000002
simd_vector simd_log(simd_vector x); // output equal to logf()
simd_vector simd_exp(simd_vector x); // output equal to expf()


//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_sin(simd_vector x)
{
    simd_vector sinus, cosinus;
    simd_sincos(x, &sinus, &cosinus);
    return sinus;
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_cos(simd_vector x)
{
    simd_vector sinus, cosinus;
    simd_sincos(x, &sinus, &cosinus);
    return cosinus;
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
