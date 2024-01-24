#ifndef __SIMD__COLOR__H__
#define __SIMD__COLOR__H__

#include "../simd.h"

//----------------------------------------------------------------------------------------------------------------------
//
// Color manipulation functions

static simd_vector simd_approx_srgb_to_linear(simd_vector value); // max error : 0.000079
static simd_vector simd_approx_linear_to_srgb(simd_vector value); // max error : 0.003851 (enough for 8bits value)


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
    simd_vector result1 = simd_polynomial6(value, (float[]) {0.10174202f, -0.38574016f, 0.85166866f, 0.45003572f, -0.018100243f, 0.00043660367f});
    return simd_select(result0, result1, big_value);
}

// TODO:
// hsv to rgb based on https://github.com/stolk/hsvbench/blob/main/hsv.h

#endif