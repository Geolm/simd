# simd
Neon/AVX simd library

This is not a math library, this a multiplatform simd intrinsic "vector size agnostic" library. There are already libraries to translate intrinsics like [SSE2Neon](https://github.com/DLTcollab/sse2neon) for example. The idea behind this library is simple : with the same code I want to use 256 bits AVX on my intel-based computer and 128 bits Neon on my M1 Mac. 


# documentation

The idea of the libray is to not assume a specific simd vector width (4 for SSE/Neon, 8 for AVX and so on) but use **simd_vector_width** variable instead. As a result the library does not contains function to shuffle lanes, horizontal operation (which are not optimal anyway). The API is based on AVX and Neon, this is not a direct AVX to Neon translation or vice versa.


simd_vector is typedef to the native simd vector of the platform (avx or neon).


## load/store/set

```C

// returns a vector loaded for the pointer [array] of floats
simd_vector simd_load(const float* array);

// loads a partial array of [count] floats, fills the rest with [unload_value]
simd_vector simd_load_partial(const float* array, int count, float unload_value);

// stores a vector [a] at the pointer [array] 
void simd_store(float* array, simd_vector a);

// stores [count] floats from vector [a] at pointer [array]
void simd_store_partial(float* array, simd_vector a, int count);

// loads 2 channels data from [array] and deinterleave data in [x] and [y].
// reads simd_vector_width*2 floats. preserves order.
void simd_load_xy(const float* array, simd_vector* x, simd_vector* y);

// loads 3 channels data from [array] and deinterleave data in [x], [y] and [z].
// reads simd_vector_width*3 floats. preserves order.
void simd_load_xyz(const float* array, simd_vector* x, simd_vector* y, simd_vector* z);

// returns a vector with all lanes set to [value] 
simd_vector simd_splat(float value);

// returns a vector with all lanes set zero
simd_vector simd_splat_zero(void);

```

## arithmetic 

```C
// returns a+b
simd_vector simd_add(simd_vector a, simd_vector b);

// returns a-b
simd_vector simd_sub(simd_vector a, simd_vector b);

// returns a*b
simd_vector simd_mul(simd_vector a, simd_vector b);

// returns a/b
simd_vector simd_div(simd_vector a, simd_vector b);

// returns 1/b
simd_vector simd_rcp(simd_vector a);

// returns 1/sqrt(a)
simd_vector simd_rsqrt(simd_vector a);

// returns sqrt(a)
simd_vector simd_sqrt(simd_vector a);

// returns the absolute value of a
simd_vector simd_abs(simd_vector a);

// returns a*b+c (fused multiply-add)
simd_vector simd_fmad(simd_vector a, simd_vector b, simd_vector c);

// returns -a
simd_vector simd_neg(simd_vector a);

```

## comparison

```C

simd_vector simd_cmp_gt(simd_vector a, simd_vector b);
simd_vector simd_cmp_ge(simd_vector a, simd_vector b); 
simd_vector simd_cmp_lt(simd_vector a, simd_vector b); 
simd_vector simd_cmp_le(simd_vector a, simd_vector b); 
simd_vector simd_cmp_eq(simd_vector a, simd_vector b); 

// returns a vector with value from a or b depending of the mask
// mask can be obtain by a comparison
simd_vector simd_select(simd_vector a, simd_vector b, simd_vector mask)

```

## rounding

```C

// returns the fractionnal part of [a]
simd_vector simd_fract(simd_vector a);

// returns the largest integer not greater than [a]
simd_vector simd_floor(simd_vector a);

// returns the smallest integer less than [a]
simd_vector simd_ceil(simd_vector a);

```

## logical

```C
// returns a logical or b (based on bits)
simd_vector simd_or(simd_vector a, simd_vector b);

// returns a logical and b (based on bits)
simd_vector simd_and(simd_vector a, simd_vector b);

// returns a logical and not b (based on bits)
simd_vector simd_andnot(simd_vector a, simd_vector b);

```

## misc

```C

// returns a sorted vector in ascending order
simd_vector simd_sort(simd_vector input);

// reverses the order of the vector
simd_vector simd_reverse(simd_vector a);

// returns a vector with minimum values from a and b
simd_vector simd_min(simd_vector a, simd_vector b);

// returns a vector with maximum values from a and b
simd_vector simd_max(simd_vector a, simd_vector b);

// returns a vector with values clamped between [range_min] and [range_max]
simd_vector simd_clamp(simd_vector a, simd_vector range_min, simd_vector range_max);

// returns a vector with values in [0;1]
simd_vector simd_saturate(simd_vector a);

// returns a linear interpolated vector based on [a] and [b] with [t] values in [0;1] range
simd_vector simd_lerp(simd_vector a, simd_vector b, simd_vector t);

```
