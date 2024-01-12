#ifndef __SIMD_MATH__H__
#define __SIMD_MATH__H__

#include "simd.h"

//----------------------------------------------------------------------------------------------------------------------
#ifndef SIMD_MATH_CONSTANTS
    #define SIMD_MATH_CONSTANTS
    #define SIMD_MATH_TAU (6.28318530f)
    #define SIMD_MATH_PI  (3.14159265f)
    #define SIMD_MATH_PI2 (1.57079632f)
    #define SIMD_MATH_PI4 (0.78539816f)
#endif

//----------------------------------------------------------------------------------------------------------------------
// prototypes
static simd_vector simd_sin(simd_vector x); // outputs same value than sinf()
static simd_vector simd_cos(simd_vector a); // outputs same value than cosf()
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c); // outputs same value than sinf()/cosf()
simd_vector simd_acos(simd_vector x); // outputs same value than acosf()
simd_vector simd_asin(simd_vector x); // outputs same value than asinf()
simd_vector simd_atan(simd_vector x); // max error with input [-10; 10]: 0.000002
simd_vector simd_atan2(simd_vector x, simd_vector y); // max error : 2.38418579E-7
simd_vector simd_log(simd_vector x); // outputs same value than logf()
simd_vector simd_exp(simd_vector x); // outputs same value than expf()


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



#endif
