#ifndef __SIMD__H__
#define __SIMD__H__

/*

    NEON/AVX simd library

    This is not a math library, this a multiplatform simd intrinsic "vector size agnostic" library. 
    With the same code it will use 256 bits AVX on intel-based computers or 128 bits NEON on arm. 

    Documentation can be found https://github.com/Geolm/simd/
*/



#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

#if _MSC_VER
#include <malloc.h>
#endif

//----------------------------------------------------------------------------------------------------------------------
// NEON
//----------------------------------------------------------------------------------------------------------------------

#if defined(__ARM_NEON) && defined(__ARM_NEON__)

#include <arm_neon.h>

#define simd_vector_width (4)
#define simd_vector_alignment (16)
typedef float32x4_t simd_vector;


//----------------------------------------------------------------------------------------------------------------------
// simd public functions
//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector simd_add(simd_vector a, simd_vector b) {return vaddq_f32(a, b);}
static inline simd_vector simd_sub(simd_vector a, simd_vector b) {return vsubq_f32(a, b);}
static inline simd_vector simd_mul(simd_vector a, simd_vector b) {return vmulq_f32(a, b);}
static inline simd_vector simd_div(simd_vector a, simd_vector b) {return vdivq_f32(a, b);}
static inline simd_vector simd_rcp(simd_vector a) {simd_vector out = vrecpeq_f32(a); return vmulq_f32(out, vrecpsq_f32(out, a));}
static inline simd_vector simd_rsqrt(simd_vector a) {simd_vector out = vrsqrteq_f32(a); return vmulq_f32(out, vrsqrtsq_f32(vmulq_f32(a, out), out));}
static inline simd_vector simd_sqrt(simd_vector a) {return vsqrtq_f32(a);}
static inline simd_vector simd_abs(simd_vector a) {return vabsq_f32(a);}
static inline simd_vector simd_abs_diff(simd_vector a, simd_vector b) {return vabdq_f32(a, b);}
static inline simd_vector simd_fmad(simd_vector a, simd_vector b, simd_vector c) {return vfmaq_f32(c, a, b);}
static inline simd_vector simd_neg(simd_vector a) {return vnegq_f32(a);}
static inline simd_vector simd_or(simd_vector a, simd_vector b) {return vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));}
static inline simd_vector simd_and(simd_vector a, simd_vector b) {return vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));}
static inline simd_vector simd_andnot(simd_vector a, simd_vector b) {return vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));}
static inline simd_vector simd_min(simd_vector a, simd_vector b) {return vminq_f32(a, b);}
static inline simd_vector simd_max(simd_vector a, simd_vector b) {return vmaxq_f32(a, b);}
static inline simd_vector simd_cmp_gt(simd_vector a, simd_vector b) {return vreinterpretq_f32_u32(vcgtq_f32(a, b));}
static inline simd_vector simd_cmp_ge(simd_vector a, simd_vector b) {return vreinterpretq_f32_u32(vcgeq_f32(a, b));}
static inline simd_vector simd_cmp_lt(simd_vector a, simd_vector b) {return vreinterpretq_f32_u32(vcltq_f32(a, b));}
static inline simd_vector simd_cmp_le(simd_vector a, simd_vector b) {return vreinterpretq_f32_u32(vcleq_f32(a, b));}
static inline simd_vector simd_cmp_eq(simd_vector a, simd_vector b) {return vreinterpretq_f32_u32(vceqq_f32(a, b));}
static inline simd_vector simd_cmp_neq(simd_vector a, simd_vector b) {return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a, b)));}
static inline simd_vector simd_select(simd_vector a, simd_vector b, simd_vector mask) {return vbslq_f32(vreinterpretq_u32_f32(mask), b, a);}
static inline simd_vector simd_reverse(simd_vector a) {return __builtin_shufflevector(a, a, 3, 2, 1, 0);}
static inline simd_vector simd_splat(float value) {return vdupq_n_f32(value);}
static inline simd_vector simd_splat_zero(void) {return vdupq_n_f32(0);}
static inline simd_vector simd_fract(simd_vector a) {return simd_sub(a, vrndq_f32(a));}
static inline simd_vector simd_floor(simd_vector a) {return vrndmq_f32(a);}
static inline simd_vector simd_ceil(simd_vector a) {return vrndpq_f32(a);}
static inline simd_vector simd_round(simd_vector a) {return vrndq_f32(a);}
static inline simd_vector simd_load(const float* array) {return vld1q_f32(array);}
static inline void simd_store(float* array, simd_vector a) {vst1q_f32(array, a);}
static inline simd_vector simd_load_partial(const float* array, int count, float unload_value)
{
    if (count >= simd_vector_width)
        return vld1q_f32(array);
    
    float32x4_t result = vsetq_lane_f32(array[0], vmovq_n_f32(unload_value), 0);
    
    if (count > 1)
        result = vsetq_lane_f32(array[1], result, 1);
    
    if (count > 2)
        result = vsetq_lane_f32(array[2], result, 2);
    
    return result;
}

static inline void simd_store_partial(float* array, simd_vector a, int count)
{   
    if (count >= simd_vector_width)
        vst1q_f32(array, a);
    else
    {
        array[0] = vgetq_lane_f32(a, 0);
        if (count > 1)
            array[1] = vgetq_lane_f32(a, 1);
        
        if (count > 2)
            array[2] = vgetq_lane_f32(a, 2);
    }
}

static inline void simd_interlace_xy(simd_vector x, simd_vector y, simd_vector* output0, simd_vector* output1)
{
    *output0 = vzip1q_f32(x, y);
    *output1 = vzip2q_f32(x, y);
}

static inline void simd_deinterlace_xy(simd_vector a, simd_vector b, simd_vector* x, simd_vector* y)
{
    *x = vuzp1q_f32(a, b);
    *y = vuzp2q_f32(a, b);
}

static inline void simd_load_xy(const float* array, simd_vector* x, simd_vector* y)
{
    float32x4x2_t data = vld2q_f32(array);
    *x = data.val[0];
    *y = data.val[1];
}

static inline void simd_load_xy_unorder(const float* array, simd_vector* x, simd_vector* y) {simd_load_xy(array, x, y);}

static inline void simd_load_xyz(const float* array, simd_vector* x, simd_vector* y, simd_vector* z)
{
    float32x4x3_t data = vld3q_f32(array);
    *x = data.val[0];
    *y = data.val[1];
    *z = data.val[2];
}

static inline void simd_load_xyz_unorder(const float* array, simd_vector* x, simd_vector* y, simd_vector* z) {simd_load_xyz(array, x, y, z);}

static inline void simd_load_xyzw(const float* array, simd_vector* x, simd_vector* y, simd_vector* z, simd_vector* w)
{
    float32x4x4_t data = vld4q_f32(array);
    *x = data.val[0];
    *y = data.val[1];
    *z = data.val[2];
    *w = data.val[3];
}

#define vblendq_f32(a, b, mask) vbslq_f32(vld1q_u32(mask), b, a)
static const uint32_t mask_0xA[4] = {0, UINT32_MAX, 0, UINT32_MAX};
static const uint32_t mask_0xC[4] = {0, 0, UINT32_MAX, UINT32_MAX};

static inline simd_vector simd_sort(simd_vector input)
{
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, mask_0xA);
    }
    {
        float32x4_t perm_neigh = simd_reverse(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, mask_0xC);
    }
    {
        float32x4_t perm_neigh = vrev64q_f32(input);
        float32x4_t perm_neigh_min = vminq_f32(input, perm_neigh);
        float32x4_t perm_neigh_max = vmaxq_f32(input, perm_neigh);
        input = vblendq_f32(perm_neigh_min, perm_neigh_max, mask_0xA);
    }
    return input;
}

static inline float simd_get_lane(simd_vector a, int lane_index)
{
    assert(lane_index>=0 && lane_index<simd_vector_width);
    switch(lane_index)
    {
    case 0 : return vgetq_lane_f32(a, 0);
    case 1 : return vgetq_lane_f32(a, 1);
    case 2 : return vgetq_lane_f32(a, 2);
    default : return vgetq_lane_f32(a, 3);
    }
}

static inline float simd_get_first_lane(simd_vector a) {return vgetq_lane_f32(a, 0);}

static inline float simd_hmin(simd_vector a)
{
    return vminvq_f32(a);
}

static inline float simd_hmax(simd_vector a)
{
    return vmaxvq_f32(a);
}

static inline float simd_hsum(simd_vector a)
{
    return vaddvq_f32(a);
}

static inline int simd_get_mask(simd_vector a)
{
    static const int32x4_t shift = {0, 1, 2, 3};
    uint32x4_t tmp = vshrq_n_u32(a, 31);
    return vaddvq_u32(vshlq_u32(tmp, shift));
}

static inline int simd_any(simd_vector a)
{
    return simd_get_mask(a) != 0;
}

static inline int simd_all(simd_vector a)
{
    return vminvq_u32(a) == UINT32_MAX;
}

static inline int simd_none(simd_vector a)
{
    return vmaxvq_u32(a) == 0;
}

static inline simd_vector simd_set_mask(int mask)
{
    uint32_t v_mask[4] = 
    {
        (mask&1)   ? 0xffffffff : 0,
        (mask&2)   ? 0xffffffff : 0,
        (mask&4)   ? 0xffffffff : 0,
        (mask&8)   ? 0xffffffff : 0,
    };
    return vreinterpretq_f32_u32(vld1q_u32(v_mask));
}

static inline void simd_export_int16(simd_vector input, int16_t* output)
{
    int32x4_t tmp = vcvtq_s32_f32(input);
    vst1_s16(output, vmovn_u32(tmp));
}

static inline void simd_export_int8(simd_vector a, simd_vector b, simd_vector c, simd_vector d, int8_t* output)
{
    int16x4x4_t value_int16;
    value_int16.val[0] = vmovn_u32(vcvtq_s32_f32(a));
    value_int16.val[1] = vmovn_u32(vcvtq_s32_f32(b));
    value_int16.val[2] = vmovn_u32(vcvtq_s32_f32(c));
    value_int16.val[3] = vmovn_u32(vcvtq_s32_f32(d));

    vst1_s8(output, vqmovn_s16(vcombine_s16(value_int16.val[0], value_int16.val[1])));
    vst1_s8(output+simd_vector_width*2, vqmovn_s16(vcombine_s16(value_int16.val[2], value_int16.val[3])));
}

static inline void simd_export_uint8(simd_vector a, simd_vector b, simd_vector c, simd_vector d, uint8_t* output)
{
    uint16x4x4_t value_uint16;
    value_uint16.val[0] = vmovn_u32(vcvtq_s32_f32(a));
    value_uint16.val[1] = vmovn_u32(vcvtq_s32_f32(b));
    value_uint16.val[2] = vmovn_u32(vcvtq_s32_f32(c));
    value_uint16.val[3] = vmovn_u32(vcvtq_s32_f32(d));

    vst1_u8(output, vqmovn_u16(vcombine_u16(value_uint16.val[0], value_uint16.val[1])));
    vst1_u8(output+simd_vector_width*2, vqmovn_u16(vcombine_u16(value_uint16.val[2], value_uint16.val[3])));
}

#else

//----------------------------------------------------------------------------------------------------------------------
// AVX
//----------------------------------------------------------------------------------------------------------------------

#include <immintrin.h>

#define simd_vector_width (8)
#define simd_vector_alignment (32)
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

// swap the two 128 bits part of the __m256
static inline __m256 _mm256_swap(__m256 a) {return _mm256_permute2f128_ps(a, a, 1);}

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
static inline simd_vector simd_neg(simd_vector a) {return _mm256_sub_ps(_mm256_setzero_ps(), a);}
static inline simd_vector simd_abs(simd_vector a) {return _mm256_max_ps(a, simd_neg(a));}
static inline simd_vector simd_abs_diff(simd_vector a, simd_vector b) {return simd_abs(simd_sub(a, b));}
static inline simd_vector simd_fmad(simd_vector a, simd_vector b, simd_vector c)
{
#ifdef __FMA__
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
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
static inline simd_vector simd_cmp_neq(simd_vector a, simd_vector b) {return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);}
static inline simd_vector simd_select(simd_vector a, simd_vector b, simd_vector mask) {return _mm256_blendv_ps(a, b, mask);}
static inline simd_vector simd_reverse(simd_vector a) {return _mm256_permute_ps(_mm256_swap(a), _MM_SHUFFLE(0, 1, 2, 3));}
static inline simd_vector simd_splat(float value) {return _mm256_set1_ps(value);}
static inline simd_vector simd_splat_zero(void) {return _mm256_setzero_ps();}
static inline simd_vector simd_fract(simd_vector a) {return simd_sub(a, _mm256_round_ps(a, _MM_FROUND_TRUNC));}
static inline simd_vector simd_round(simd_vector a) {return _mm256_round_ps(a, _MM_FROUND_NINT);}
static inline simd_vector simd_floor(simd_vector a) {return _mm256_floor_ps(a);}
static inline simd_vector simd_ceil(simd_vector a) {return _mm256_ceil_ps(a);}
static inline simd_vector simd_load(const float* array)
{
    if ((((uintptr_t)(array)) & 31 ) == 0) // aligned on 32 bytes
        return _mm256_load_ps(array);
    
    return _mm256_loadu_ps(array);
}
static inline void simd_store(float* array, simd_vector a)
{
    if ((((uintptr_t)(array)) & 31 ) == 0) // aligned on 32 bytes
        _mm256_store_ps(array, a);
    else
        _mm256_storeu_ps(array, a);
}
static inline simd_vector simd_load_partial(const float* array, int count, float unload_value)
{
    assert(count>0);
    if (count >= simd_vector_width)
        return simd_load(array);
    
    __m256 inf_mask = _mm256_set_ps(unload_value, (count>6) ? 0.f : unload_value, (count>5) ? 0.f : unload_value, (count>4) ? 0.f : unload_value,
                                    (count>3) ? 0.f : unload_value, (count>2) ? 0.f : unload_value, (count>1) ? 0.f : unload_value, (count>0) ? 0.f : unload_value);
    
    __m256 a = _mm256_maskload_ps(array, loadstore_mask(count));
    return _mm256_or_ps(a, inf_mask);
}

static inline void simd_store_partial(float* array, simd_vector a, int count)
{
    assert(count>0);
    if (count >= simd_vector_width)
        simd_store(array, a);
    else
        _mm256_maskstore_ps(array, loadstore_mask(count), a);
}

static inline void simd_interlace_xy(simd_vector x, simd_vector y, simd_vector* output0, simd_vector* output1)
{
    __m128 x_lo = _mm256_extractf128_ps(x, 0);
    __m128 x_hi = _mm256_extractf128_ps(x, 1);
    __m128 y_lo = _mm256_extractf128_ps(y, 0);
    __m128 y_hi = _mm256_extractf128_ps(y, 1);

    *output0 = _mm256_set_m128(_mm_unpackhi_ps(x_lo, y_lo), _mm_unpacklo_ps(x_lo, y_lo));
    *output1 = _mm256_set_m128(_mm_unpackhi_ps(x_hi, y_hi), _mm_unpacklo_ps(x_hi, y_hi));
}

static inline void simd_deinterlace_xy(simd_vector a, simd_vector b, simd_vector* x, simd_vector* y)
{
    *x = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    *y = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));

    // do additionnal shuffle to preserve order
    simd_vector tmp;
    tmp = _mm256_swap(*x);
    tmp = _mm256_permute_ps(tmp, _MM_SHUFFLE(1, 0, 3, 2));
    *x = _mm256_blend_ps(*x, tmp, 0x3C);   // 00111100b = 0x3C
    
    tmp = _mm256_swap(*y);
    tmp = _mm256_permute_ps(tmp, _MM_SHUFFLE(1, 0, 3, 2));
    *y = _mm256_blend_ps(*y, tmp, 0x3C);   // 00111100b = 0x3C
}

static inline void simd_load_xy_unorder(const float* array, simd_vector* x, simd_vector* y)
{
    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);
    
    *x = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    *y = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
}

static inline void simd_load_xy(const float* array, simd_vector* x, simd_vector* y)
{
    simd_deinterlace_xy(simd_load(array), simd_load(array + simd_vector_width), x, y);
}

static inline void simd_load_xyz_unorder(const float* array, simd_vector* x, simd_vector* y, simd_vector* z)
{
    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);
    simd_vector c = simd_load(array + simd_vector_width * 2);

    simd_vector tmp = _mm256_blend_ps(a, b, 0x92);  // 01001001b = 0x92 (intel reverse order)
    *x = _mm256_blend_ps(tmp, c, 0x24); // 00100100b = 0x24

    tmp = _mm256_blend_ps(a, b, 0x24);
    *y = _mm256_blend_ps(tmp, c, 0x49);     // 10010010b = 0x49 (intel reverse order)

    tmp = _mm256_blend_ps(a, b, 0x49);
    *z = _mm256_blend_ps(tmp, c, 0x92);
}

static inline void simd_load_xyz(const float* array, simd_vector* x, simd_vector* y, simd_vector* z)
{
    simd_load_xyz_unorder(array, x, y, z);
    
    // do additionnal shuffle to preserve order
    *x = _mm256_permute_ps(*x, _MM_SHUFFLE(1, 2, 3, 0));
    *x = _mm256_blend_ps(*x, _mm256_swap(*x), 0x44);   // 00100010b = 0x44 (intel reverse order)
    *y = _mm256_permute_ps(*y, _MM_SHUFFLE(2, 3, 0, 1));
    *y = _mm256_blend_ps(*y, _mm256_swap(*y), 0x66);   // 01100110b = 0x66
    *z = _mm256_permute_ps(*z, _MM_SHUFFLE(3, 0, 1, 2));
    *z = _mm256_blend_ps(*z, _mm256_swap(*z), 0x22);   // 01000100b = 0x22 (intel reverse order)
}

static inline void simd_load_xyzw_unorder(const float* array, simd_vector* x, simd_vector* y, simd_vector* z, simd_vector* w)
{
    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);
    simd_vector c = simd_load(array + simd_vector_width * 2);
    simd_vector d = simd_load(array + simd_vector_width * 3);

    b = _mm256_permute_ps(b, _MM_SHUFFLE(2, 1, 0, 3));
    c = _mm256_permute_ps(c, _MM_SHUFFLE(1, 0, 3, 2));
    d = _mm256_permute_ps(d, _MM_SHUFFLE(0, 3, 2, 1));

    *x = _mm256_blend_ps(a, b, 0x22);    // 01000100b = 0x22 (intel reverse order)
    *x = _mm256_blend_ps(*x, c, 0x44);   // 00100010b = 0x44 (intel reverse order)
    *x = _mm256_blend_ps(*x, d, 0x88);   // 00010001b = 0x88 (intel reverse order)

    *y = _mm256_blend_ps(d, a, 0x22);
    *y = _mm256_blend_ps(*y, b, 0x44);
    *y = _mm256_blend_ps(*y, c, 0x88);

    *z = _mm256_blend_ps(c, d, 0x22);
    *z = _mm256_blend_ps(*z, a, 0x44);
    *z = _mm256_blend_ps(*z, b, 0x88);

    *w = _mm256_blend_ps(b, c, 0x22);
    *w = _mm256_blend_ps(*w, d, 0x44);
    *w = _mm256_blend_ps(*w, a, 0x88);
}

static inline void simd_load_xyzw(const float* array, simd_vector* x, simd_vector* y, simd_vector* z, simd_vector* w)
{
    simd_load_xyzw_unorder(array, x, y, z, w);
    
    simd_vector swap = _mm256_swap(*x);
    simd_vector lo = _mm256_unpacklo_ps(*x, swap);
    simd_vector hi = _mm256_unpackhi_ps(*x, swap);
    *x = _mm256_permute2f128_ps(lo, hi, 2<<4);
    
    *y = _mm256_permute_ps(*y, _MM_SHUFFLE(0, 3, 2, 1));
    swap = _mm256_swap(*y);
    lo = _mm256_unpacklo_ps(*y, swap);
    hi = _mm256_unpackhi_ps(*y, swap);
    *y = _mm256_permute2f128_ps(lo, hi, 0x20);
    
    *z = _mm256_permute_ps(*z, _MM_SHUFFLE(1, 0, 3, 2));
    swap = _mm256_swap(*z);
    lo = _mm256_unpacklo_ps(*z, swap);
    hi = _mm256_unpackhi_ps(*z, swap);
    *z = _mm256_permute2f128_ps(lo, hi, 0x20);
    
    *w = _mm256_permute_ps(*w, _MM_SHUFFLE(2, 1, 0, 3));
    swap = _mm256_swap(*w);
    lo = _mm256_unpacklo_ps(*w, swap);
    hi = _mm256_unpackhi_ps(*w, swap);
    *w = _mm256_permute2f128_ps(lo, hi, 0x20);
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

static inline float simd_get_lane(simd_vector a, int lane_index)
{
    assert(lane_index>=0 && lane_index<simd_vector_width);

    union __mm256tofloat 
    {
        __m256 v;
        float f[8];
    };

    const union __mm256tofloat u = { a };
    return u.f[lane_index];
}

static inline float simd_get_first_lane(simd_vector a) {return _mm256_cvtss_f32(a);}

static inline float simd_hmin(simd_vector a)
{
    a = simd_min(a, _mm256_permute_ps(a, _MM_SHUFFLE(2, 1, 0, 3)));
    a = simd_min(a, _mm256_permute_ps(a, _MM_SHUFFLE(1, 0, 3, 2)));
    a = simd_min(a, _mm256_swap(a));
    return simd_get_first_lane(a);
}

static inline float simd_hmax(simd_vector a)
{
    a = simd_max(a, _mm256_permute_ps(a, _MM_SHUFFLE(2, 1, 0, 3)));
    a = simd_max(a, _mm256_permute_ps(a, _MM_SHUFFLE(1, 0, 3, 2)));
    a = simd_max(a, _mm256_swap(a));
    return simd_get_first_lane(a);
}

static inline float simd_hsum(simd_vector a)
{
    a = simd_add(a, _mm256_permute_ps(a, _MM_SHUFFLE(2, 1, 0, 3)));
    a = simd_add(a, _mm256_permute_ps(a, _MM_SHUFFLE(1, 0, 3, 2)));
    a = simd_add(a, _mm256_swap(a));
    return simd_get_first_lane(a);
}

static inline int simd_get_mask(simd_vector a)
{
    return _mm256_movemask_ps(a);
}

static inline int simd_any(simd_vector a)
{
    return _mm256_movemask_ps(a) != 0;
}

static inline int simd_all(simd_vector a)
{
    return _mm256_movemask_ps(a) == 0xff;
}

static inline int simd_none(simd_vector a)
{
    return _mm256_movemask_ps(a) == 0;
}

static inline simd_vector simd_set_mask(int mask)
{
    return _mm256_cvtepi32_ps(_mm256_set_epi32(
        (mask&128) ? 0xffffffff : 0,
        (mask&64)  ? 0xffffffff : 0,
        (mask&32)  ? 0xffffffff : 0,
        (mask&16)  ? 0xffffffff : 0,
        (mask&8)   ? 0xffffffff : 0,
        (mask&4)   ? 0xffffffff : 0,
        (mask&2)   ? 0xffffffff : 0,
        (mask&1)   ? 0xffffffff : 0));
}

// convert float to int16_t : it's up to the caller to be sure the float are in the right range [-32768.f; 32767.f]
static inline void simd_export_int16(simd_vector input, int16_t* output)
{
    __m256i tmp = _mm256_cvtps_epi32(input);
    tmp = _mm256_packs_epi32(tmp, _mm256_setzero_si256());
    tmp = _mm256_permute4x64_epi64(tmp, 0xD8);
    _mm_storeu_si128((__m128i*)output, _mm256_castsi256_si128(tmp));
}

static inline void simd_export_int8(simd_vector a, simd_vector b, simd_vector c, simd_vector d, int8_t* output)
{
    __m256i ab_int16 = _mm256_packs_epi32(_mm256_cvtps_epi32(a), _mm256_cvtps_epi32(c));
    __m256i cd_int16 = _mm256_packs_epi32(_mm256_cvtps_epi32(b), _mm256_cvtps_epi32(d));

    ab_int16 = _mm256_permute4x64_epi64(ab_int16, 0xD8);
    cd_int16 = _mm256_permute4x64_epi64(cd_int16, 0xD8);

    __m256i pack = _mm256_packs_epi16(ab_int16, cd_int16);
    _mm256_store_si256((__m256i*)output, pack);
}

static inline void simd_export_uint8(simd_vector a, simd_vector b, simd_vector c, simd_vector d, uint8_t* output)
{
    // preping the float
    simd_vector threshold = simd_splat(127.f);
    simd_vector c2 = simd_splat(-256.f);

    a = simd_select(a, simd_add(c2, a), simd_cmp_ge(a, threshold));
    b = simd_select(b, simd_add(c2, b), simd_cmp_ge(b, threshold));
    c = simd_select(c, simd_add(c2, c), simd_cmp_ge(c, threshold));
    d = simd_select(d, simd_add(c2, d), simd_cmp_ge(d, threshold));

    simd_export_int8(a, b, c, d, (int8_t*)output);
}

#endif

//----------------------------------------------------------------------------------------------------------------------
// common public functions
//----------------------------------------------------------------------------------------------------------------------

#define simd_last_lane (simd_vector_width-1)
#define simd_full_mask ((1<<simd_vector_width)-1)

static inline simd_vector simd_clamp(simd_vector a, simd_vector range_min, simd_vector range_max) {return simd_max(simd_min(a, range_max), range_min);}
static inline simd_vector simd_saturate(simd_vector a) {return simd_clamp(a, simd_splat_zero(), simd_splat(1.f));}
static inline simd_vector simd_lerp(simd_vector a, simd_vector b, simd_vector t) {return simd_fmad(simd_sub(a, b), t, a);}

//-----------------------------------------------------------------------------
static inline simd_vector simd_isnan(simd_vector a)
{
    return simd_cmp_neq(a, a);
}

//-----------------------------------------------------------------------------
static inline simd_vector simd_equal(simd_vector a, simd_vector b, simd_vector epsilon)
{
    simd_vector diff = simd_abs_diff(a, b);
    return simd_cmp_lt(diff, epsilon);
}

//-----------------------------------------------------------------------------
static inline void* simd_aligned_alloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, simd_vector_alignment);
#else
    size = (size + simd_vector_alignment - 1) / simd_vector_alignment;
    size *= simd_vector_alignment;
    return aligned_alloc(simd_vector_alignment, size);
#endif
}

//-----------------------------------------------------------------------------
static inline void simd_aligned_free(void* ptr)
{
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

//-----------------------------------------------------------------------------
static inline simd_vector simd_sign(simd_vector a)
{
    simd_vector result = simd_select(simd_splat_zero(), simd_splat(-1.f), simd_cmp_lt(a, simd_splat_zero()));
    return simd_select(result, simd_splat( 1.f), simd_cmp_gt(a, simd_splat_zero()));
}

//-----------------------------------------------------------------------------
static inline void simd_interlace_xyzw(simd_vector x, simd_vector y, simd_vector z, simd_vector w,
                                       simd_vector* output0, simd_vector* output1, simd_vector* output2, simd_vector* output3)
{
    simd_vector xz0, xz1;
    simd_interlace_xy(x, z, &xz0, &xz1);

    simd_vector yw0, yw1;
    simd_interlace_xy(y, w, &yw0, &yw1);

    simd_interlace_xy(xz0, yw0, output0, output1);
    simd_interlace_xy(xz1, yw1, output2, output3);
}

//-----------------------------------------------------------------------------
static inline void simd_export_color(simd_vector red, simd_vector green, simd_vector blue, simd_vector alpha, uint8_t* output)
{
    simd_vector rgba0, rgba1, rgba2, rgba3;
    simd_interlace_xyzw(red, green, blue, alpha, &rgba0, &rgba1, &rgba2, &rgba3);

    simd_export_uint8(rgba0, rgba1, rgba2, rgba3, output);
}



#endif // __SIMD__H__
