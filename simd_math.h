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
static simd_vector simd_sin(simd_vector x); // max error : 5.960464478e-08
static simd_vector simd_cos(simd_vector a); // max error : 5.960464478e-08
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c); // max error : 5.960464478e-08
simd_vector simd_acos(simd_vector x); // max error : 2.384185791e-07
simd_vector simd_asin(simd_vector x); // max error : 1.192092896e-07
simd_vector simd_atan(simd_vector x); // max error : 6.699562073e-05
simd_vector simd_atan2(simd_vector x, simd_vector y); // max error : 2.384185791e-07
simd_vector simd_log(simd_vector x); // max error : 4.768371582e-07
simd_vector simd_exp(simd_vector x); // max error : 1.108270880e-07
simd_vector simd_cbrt(simd_vector x); // max error : 4.768371582e-07


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
