#ifndef __SIMD_VEC2__H__
#define __SIMD_VEC2__H__

#include "simd.h"
#include <stddef.h>


// SoA, Structure of Array, Vec2
typedef struct
{
    simd_vector x;
    simd_vector y;
} simd_vec2;


static inline simd_vec2 simd_vec2_load(const float* x, const float* y, ptrdiff_t offset) {return (simd_vec2) {.x = simd_load(x + offset), .y = simd_load(y + offset)};}
static inline simd_vec2 simd_vec2_add(simd_vec2 a, simd_vec2 b) {return (simd_vec2) {.x = simd_add(a.x, b.x), .y = simd_add(a.y, b.y)};}
static inline simd_vec2 simd_vec2_sub(simd_vec2 a, simd_vec2 b) {return (simd_vec2) {.x = simd_sub(a.x, b.x), .y = simd_sub(a.y, b.y)};}
static inline simd_vec2 simd_vec2_mul(simd_vec2 a, simd_vec2 b) {return (simd_vec2) {.x = simd_mul(a.x, b.x), .y = simd_mul(a.y, b.y)};}
static inline simd_vec2 simd_vec2_clamp(simd_vec2 a, simd_vec2 range_min, simd_vec2 range_max) 
{
    return (simd_vec2) {.x = simd_clamp(a.x, range_min.x, range_max.x), .y = simd_clamp(a.y, range_min.y, range_max.y)};
}
static inline simd_vector simd_vec2_sq_length(simd_vec2 a) {return simd_fmad(a.x, a.x, simd_mul(a.y, a.y));}
static inline simd_vector simd_vec2_dot(simd_vec2 a, simd_vec2 b) {return simd_fmad(a.x, b.x, simd_mul(a.y, b.y));}


#endif