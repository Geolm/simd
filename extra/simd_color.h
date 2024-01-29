#ifndef __SIMD__COLOR__H__
#define __SIMD__COLOR__H__

#include "simd_math.h"

//----------------------------------------------------------------------------------------------------------------------
//
// Color manipulation functions

static simd_vector simd_srgb_to_linear(simd_vector value); // max error : 4.267692566e-05
static simd_vector simd_linear_to_srgb(simd_vector value); // max error : 1.192092896e-07


//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_linear_to_srgb(simd_vector value)
{
    simd_vector small_value = simd_cmp_lt(value, simd_splat(0.0031308f));
    simd_vector result_small = simd_mul(value, simd_splat(12.92f));
    simd_vector value_1_3 = simd_cbrt(value);
    simd_vector result = simd_mul(value_1_3, simd_sqrt(simd_sqrt(value_1_3)));
    result = simd_fmad(simd_splat(1.055f), result, simd_splat(-0.055f));
    result = simd_select(result, result_small, small_value);
    return result;
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_srgb_to_linear(simd_vector value)
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

