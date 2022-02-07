#ifndef __SIMD__H__
#define __SIMD__H__


// positive infinity float hexadecimal value
#define simd_float_p_infinity (0x7F800000)

//----------------------------------------------------------------------------------------------------------------------
// Neon
//----------------------------------------------------------------------------------------------------------------------

#if defined(__ARM_NEON) && defined(__ARM_NEON__)

#include <arm_neon.h>

#define simd_vector_width (4)

typedef float32x4_t simd_vector;


//----------------------------------------------------------------------------------------------------------------------
// simd public functions
//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_add(simd_vector a, simd_vector b) {return vaddq_f32(a, b);}
static inline simd_vector simd_sub(simd_vector a, simd_vector b) {return vsubq_f32(a, b);}
static inline simd_vector simd_mul(simd_vector a, simd_vector b) {return vmulq_f32(a, b);}
static inline simd_vector simd_div(simd_vector a, simd_vector b) {return vdivq_f32(a, b);}
static inline simd_vector simd_rcp(simd_vector a) {simd_vector recip = vrecpeq_f32(a); return vmulq_f32(recip, vrecpsq_f32(recip, a));}
static inline simd_vector simd_rsqrt(simd_vector a) {return vrsqrteq_f32(a);}
static inline simd_vector simd_sqrt(simd_vector a) {return vsqrtq_f32(a);}
static inline simd_vector simd_abs(simd_vector a) {return vabsq_f32(a);}
static inline simd_vector simd_abs_diff(simd_vector a, simd_vector b) {return vabdq_f32(a, b);}
static inline simd_vector simd_fmad(simd_vector a, simd_vector b, simd_vector c) {return vfmaq_f32(c, a, b);}
static inline simd_vector simd_neg(simd_vector a) {return vnegq_f32(a);}
static inline simd_vector simd_or(simd_vector a, simd_vector b) {return vorrq_s32(a, b);}
static inline simd_vector simd_and(simd_vector a, simd_vector b) {return vandq_s32(a, b);}
static inline simd_vector simd_andnot(simd_vector a, simd_vector b) {return vbicq_s32(a, b);}
static inline simd_vector simd_min(simd_vector a, simd_vector b) {return vminq_f32(a, b);}
static inline simd_vector simd_max(simd_vector a, simd_vector b) {return vmaxq_f32(a, b);}
static inline simd_vector simd_cmp_gt(simd_vector a, simd_vector b) {return vcgtq_f32(a, b);}
static inline simd_vector simd_cmp_ge(simd_vector a, simd_vector b) {return vcgeq_f32(a, b);}
static inline simd_vector simd_cmp_lt(simd_vector a, simd_vector b) {return vcltq_f32(a, b);}
static inline simd_vector simd_cmp_le(simd_vector a, simd_vector b) {return vcleq_f32(a, b);}
static inline simd_vector simd_cmp_eq(simd_vector a, simd_vector b) {return vceqq_f32(a, b);}
static inline simd_vector simd_select(simd_vector a, simd_vector b, simd_vector mask) {return vbslq_f32(mask, b, a);}
static inline simd_vector simd_reverse(simd_vector a) {return vrev64q_f32(a);}
static inline simd_vector simd_splat(float value) {return vdupq_n_f32(value);}
static inline simd_vector simd_splat_zero(void) {return vdupq_n_f32(0);}

static inline simd_vector simd_load(const float* array, int index) {return vld1q_f32(array + simd_vector_width * index);}
static inline void simd_store(float* array, int index, simd_vector a) {vst1q_f32(array + simd_vector_width * index, a);}
static inline simd_vector simd_load_partial(const float* array, int index, int count, uint32_t unload_value)
{
    int array_index = simd_vector_width * index;
    if (count == simd_vector_width)
        return vld1q_f32(array + array_index);
    
    float32x4_t result = vsetq_lane_f32(array[array_index], vmovq_n_f32(*(float*)&unload_value), 0);
    
    if (count > 1)
        result = vsetq_lane_f32(array[array_index + 1], result, 1);
    
    if (count > 2)
        result = vsetq_lane_f32(array[array_index + 2], result, 2);
    
    return result;
}

static inline void simd_store_partial(float* array, int index, simd_vector a, int count)
{   
    int array_index = simd_vector_width * index;
    if (count == simd_vector_width)
        vst1q_f32(array + array_index, a);
    else
    {
        array[array_index] = vgetq_lane_f32(a, 0);
        if (count > 1)
            array[array_index+1] = vgetq_lane_f32(a, 1);
        
        if (count > 2)
            array[array_index+2] = vgetq_lane_f32(a, 2);
    }
}

static inline void simd_load_xy(const float* array, simd_sector* x, simd_sector* y)
{
    float32x4x3_t data = vld2q_f32(array);
    *x = data.val[0];
    *y = data.val[1];
}

static inline void simd_load_xyz(const float* array, simd_sector* x, simd_sector* y, simd_sector* z)
{
    float32x4x3_t data = vld3q_f32(array);
    *x = data.val[0];
    *y = data.val[1];
    *z = data.val[2];
}

static inline void simd_load_xyzw(const float* array, simd_sector* x, simd_sector* y, simd_sector* z, simd_sector* w)
{
    float32x4x3_t data = vld4q_f32(array);
    *x = data.val[0];
    *y = data.val[1];
    *z = data.val[2];
    *w = data.val[3];
}

#else // NEON

//----------------------------------------------------------------------------------------------------------------------
// AVX
//----------------------------------------------------------------------------------------------------------------------

#include <immintrin.h>
#include <stdint.h>

#define simd_vector_width (8)

typedef __m256 simd_vector;


// private function, you should not use this function, specific to AVX implementation
static inline __m256i loadstore_mask(int element_count)
{
    return _mm256_set_epi32((element_count>7) ? 0xffffffff : 0, 
                            (element_count>6) ? 0xffffffff : 0,
                            (element_count>5) ? 0xffffffff : 0,
                            (element_count>4) ? 0xffffffff : 0,
                            (element_count>3) ? 0xffffffff : 0,
                            (element_count>2) ? 0xffffffff : 0,
                            (element_count>1) ? 0xffffffff : 0,
                            (element_count>0) ? 0xffffffff : 0);
}

//----------------------------------------------------------------------------------------------------------------------
// simd public functions
//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_add(simd_vector a, simd_vector b) {return _mm256_add_ps(a, b);}
static inline simd_vector simd_sub(simd_vector a, simd_vector b) {return _mm256_sub_ps(a, b);}
static inline simd_vector simd_mul(simd_vector a, simd_vector b) {return _mm256_mul_ps(a, b);}
static inline simd_vector simd_div(simd_vector a, simd_vector b) {return _mm256_div_ps(a, b);}
static inline simd_vector simd_rcp(simd_vector a) {return _mm256_rcp_ps(a);}
static inline simd_vector simd_rsqrt(simd_vector a) {return _mm256_rsqrt_ps(a);}
static inline simd_vector simd_sqrt(simd_vector a) {return _mm256_sqrt_ps(a);}
static inline simd_vector simd_abs(simd_vector a)
{
    const __m256i minus1 = _mm256_set1_epi32(-1);
    return _mm256_and_ps(a, _mm256_cvtepi32_ps(minus1));
}
static inline simd_vector simd_abs_diff(simd_vector a, simd_vector b) {return simd_abs(simd_sub(a, b));}
static inline simd_vector simd_fmad(simd_vector a, simd_vector b, simd_vector c) {return _mm256_fmadd_ps(a, b, c);}
static inline simd_vector simd_neg(simd_vector a) {return _mm256_sub_ps(_mm256_setzero_ps(), a);}
static inline simd_vector simd_load(const float* array, int index) {return _mm256_loadu_ps(array + simd_vector_width * index);}
static inline void simd_store(float* array, int index, simd_vector a) {_mm256_storeu_ps(array + simd_vector_width * index, a);}
static inline simd_vector simd_or(simd_vector a, simd_vector b) {return _mm256_or_ps(a, b);}
static inline simd_vector simd_and(simd_vector a, simd_vector b) {return _mm256_and_ps(a, b);}
static inline simd_vector simd_andnot(simd_vector a, simd_vector b) {return _mm256_andnot_ps(a, b);}
static inline simd_vector simd_min(simd_vector a, simd_vector b) {return _mm256_min_ps(a, b);}
static inline simd_vector simd_max(simd_vector a, simd_vector b) {return _mm256_max_ps(a, b);}
static inline simd_vector simd_cmp_gt(simd_vector a, simd_vector b) {return _mm256_cmp_ps(a, b, _CMP_GT_OQ);}
static inline simd_vector simd_cmp_ge(simd_vector a, simd_vector b) {return _mm256_cmp_ps(a, b, _CMP_GE_OQ);}
static inline simd_vector simd_cmp_lt(simd_vector a, simd_vector b) {return _mm256_cmp_ps(a, b, _CMP_LT_OQ);}
static inline simd_vector simd_cmp_le(simd_vector a, simd_vector b) {return _mm256_cmp_ps(a, b, _CMP_LE_OQ);}
static inline simd_vector simd_cmp_eq(simd_vector a, simd_vector b) {return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);}
static inline simd_vector simd_select(simd_vector a, simd_vector b, simd_vector mask) {return _mm256_blendv_ps(a, b, mask);}
static inline simd_vector simd_reverse(simd_vector a)
{
    __m128 lo = _mm256_extractf128_ps(a, 0);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m256 swap = _mm256_setr_m128(hi, lo);
     return _mm256_permute_ps(swap, _MM_SHUFFLE(0, 1, 2, 3));
}
static inline simd_vector simd_splat(float value) {return _mm256_set1_ps(value);}
static inline simd_vector simd_splat_zero(void) {return _mm256_setzero_ps();}


static inline simd_vector simd_load_partial(const float* array, int index, int count, uint32_t unload_value)
{
    if (count >= simd_vector_width)
        return simd_load(array, index);
    
    __m256 inf_mask = _mm256_cvtepi32_ps(
                        _mm256_set_epi32(unload_value, (count>6) ? 0 : unload_value, (count>5) ? 0 : unload_value, (count>4) ? 0 : unload_value,
                                         (count>3) ? 0 : unload_value, (count>2) ? 0 : unload_value, (count>1) ? 0 : unload_value, (count>0) ? 0 : unload_value));
    
    __m256 a = _mm256_maskload_ps(array + simd_vector_width * index, loadstore_mask(count));
    return _mm256_or_ps(a, inf_mask);
}

static inline void simd_store_partial(float* array, int index, simd_vector a, int count)
{
    if (count >= simd_vector_width)
        simd_store(array, index, a);
    else
        _mm256_maskstore_ps(array + simd_vector_width * index, loadstore_mask(count), a);
}

static inline void simd_load_xy(const float* array, simd_vector* x, simd_vector* y)
{
    simd_vector a = simd_load(array, 0);
    simd_vector b = simd_load(array, 1);

    *x = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    *y = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
}

static inline simd_vector simd_sort(simd_vector input)
{
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(0, 1, 2, 3));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    {
        __m256 perm_neigh = simd_reverse(input);
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xF0);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(1, 0, 3, 2));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 perm_neigh_min = _mm256_min_ps(input, perm_neigh);
        __m256 perm_neigh_max = _mm256_max_ps(input, perm_neigh);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    return input;
}

#endif

//----------------------------------------------------------------------------------------------------------------------
// common public functions
//----------------------------------------------------------------------------------------------------------------------

static inline simd_vector simd_clamp(simd_vector a, simd_vector range_min, simd_vector range_max) {return simd_max(simd_min(a, range_max), range_min);}
static inline simd_vector simd_saturate(simd_vector a) {return simd_clamp(a, simd_splat_zero(), simd_splat(1.f));}
static inline simd_vector simd_lerp(simd_vector a, simd_vector b, simd_vector t) {return simd_fmad(simd_sub(a, b), t, a);}

#endif // __SIMD__H__
