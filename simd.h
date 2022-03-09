#ifndef __SIMD__H__
#define __SIMD__H__

#include <assert.h>

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
static inline simd_vector simd_fract(simd_vector a) {return simd_sub(a, vrndq_f32(a));}
static inline simd_vector simd_floor(simd_vector a) {return vrndmq_f32(a);}
static inline simd_vector simd_ceil(simd_vector a) {return vrndpq_f32(a);}
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
        float32x4_t perm_neigh = __builtin_shufflevector(input, input, 3, 2, 1, 0);
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

#else

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

// swap the two 128 bits part of the __m256
static inline __m256 _mm256_swap(__m256 a) {return _mm256_permute2f128_ps(a, a, _MM_SHUFFLE(0, 0, 1, 1));}

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
static inline simd_vector simd_abs(simd_vector a) {return _mm256_and_ps(a, _mm256_cvtepi32_ps(_mm256_set1_epi32(-1)));}
static inline simd_vector simd_abs_diff(simd_vector a, simd_vector b) {return simd_abs(simd_sub(a, b));}
static inline simd_vector simd_fmad(simd_vector a, simd_vector b, simd_vector c) {return _mm256_fmadd_ps(a, b, c);}
static inline simd_vector simd_neg(simd_vector a) {return _mm256_sub_ps(_mm256_setzero_ps(), a);}
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
static inline simd_vector simd_reverse(simd_vector a) {return _mm256_permute_ps(_mm256_swap(a), _MM_SHUFFLE(0, 1, 2, 3));}
static inline simd_vector simd_splat(float value) {return _mm256_set1_ps(value);}
static inline simd_vector simd_splat_zero(void) {return _mm256_setzero_ps();}
static inline simd_vector simd_fract(simd_vector a) {return simd_sub(a, _mm256_round_ps(a, _MM_FROUND_TRUNC));}
static inline simd_vector simd_floor(simd_vector a) {return _mm256_floor_ps(a);}
static inline simd_vector simd_ceil(simd_vector a) {return _mm256_ceil_ps(a);}
static inline simd_vector simd_load(const float* array) {return _mm256_loadu_ps(array);}
static inline void simd_store(float* array, simd_vector a) {_mm256_storeu_ps(array, a);}
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

static inline void simd_load_xy_unorder(const float* array, simd_vector* x, simd_vector* y)
{
    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);
    
    *x = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    *y = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
}

static inline void simd_load_xy(const float* array, simd_vector* x, simd_vector* y)
{
    simd_load_xy_unorder(array, x, y);

    // do additionnal shuffle to preserve order
    simd_vector tmp;
    tmp = _mm256_swap(*x);
    tmp = _mm256_permute_ps(tmp, _MM_SHUFFLE(1, 0, 3, 2));
    *x = _mm256_blend_ps(*x, tmp, 0x3C);   // 00111100b = 0x3C
    
    tmp = _mm256_swap(*y);
    tmp = _mm256_permute_ps(tmp, _MM_SHUFFLE(1, 0, 3, 2));
    *y = _mm256_blend_ps(*y, tmp, 0x3C);   // 00111100b = 0x3C
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

static inline void simd_load_xyzw(const float* array, simd_vector* x, simd_vector* y, simd_vector* z, simd_vector* w)
{
    
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

/*
static inline simd_vector simd_sin(simd_vector a)
{
    

    // Uses a minimax polynomial fitted to the [-pi/2, pi/2] range
	inline n128 _hlslpp_sin_ps(n128 x)
	{
		static const n128 sin_c1 = f4_1;
		static const n128 sin_c3 = _hlslpp_set1_ps(-1.6665578e-1f);
		static const n128 sin_c5 = _hlslpp_set1_ps(8.3109378e-3f);
		static const n128 sin_c7 = _hlslpp_set1_ps(-1.84477486e-4f);

		// Range reduction (into [-pi, pi] range)
		// Formula is x = x - round(x / 2pi) * 2pi

		x = _hlslpp_subm_ps(x, _hlslpp_round_ps(_hlslpp_mul_ps(x, f4_inv2pi)), f4_2pi);

		n128 gtpi2 = _hlslpp_cmpgt_ps(x, f4_pi2);
		n128 ltminusPi2 = _hlslpp_cmplt_ps(x, f4_minusPi2);

		n128 ox = x;

		// Use identities/mirroring to remap into the range of the minimax polynomial
		x = _hlslpp_sel_ps(x, _hlslpp_sub_ps(f4_pi, ox), gtpi2);
		x = _hlslpp_sel_ps(x, _hlslpp_sub_ps(f4_minusPi, ox), ltminusPi2);

		n128 x2 = _hlslpp_mul_ps(x, x);
		n128 result;
		result = _hlslpp_madd_ps(x2, sin_c7, sin_c5);
		result = _hlslpp_madd_ps(x2, result, sin_c3);
		result = _hlslpp_madd_ps(x2, result, sin_c1);
		result = _hlslpp_mul_ps(result, x);
		return result;
	}

    
}
 */

#endif

//----------------------------------------------------------------------------------------------------------------------
// common public functions
//----------------------------------------------------------------------------------------------------------------------

static inline simd_vector simd_clamp(simd_vector a, simd_vector range_min, simd_vector range_max) {return simd_max(simd_min(a, range_max), range_min);}
static inline simd_vector simd_saturate(simd_vector a) {return simd_clamp(a, simd_splat_zero(), simd_splat(1.f));}
static inline simd_vector simd_lerp(simd_vector a, simd_vector b, simd_vector t) {return simd_fmad(simd_sub(a, b), t, a);}


#endif // __SIMD__H__
