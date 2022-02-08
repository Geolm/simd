# simd
Neon/AVX simd library, vector size agnostic


# documentation

The idea of the libray is to not assume a specific simd vector width (4 for SSE/Neon, 8 for AVX and so on) but use **simd_vector_width** variable instead. As a result the library does not contains function to get lanes value, nor do shuffle based on the width of the vector. The API is based on AVX and Neon, some functions are composed of multiple simd instructions.


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

simd_vector simd_add(simd_vector a, simd_vector b);
simd_vector simd_sub(simd_vector a, simd_vector b);
simd_vector simd_mul(simd_vector a, simd_vector b);
simd_vector simd_div(simd_vector a, simd_vector b);
simd_vector simd_rcp(simd_vector a);
simd_vector simd_rsqrt(simd_vector a);
simd_vector simd_sqrt(simd_vector a);
simd_vector simd_abs(simd_vector a);
simd_vector simd_fmad(simd_vector a, simd_vector b, simd_vector c);
simd_vector simd_neg(simd_vector a);

```

## comparison

```C

simd_vector simd_cmp_gt(simd_vector a, simd_vector b);
simd_vector simd_cmp_ge(simd_vector a, simd_vector b); 
simd_vector simd_cmp_lt(simd_vector a, simd_vector b); 
simd_vector simd_cmp_le(simd_vector a, simd_vector b); 
simd_vector simd_cmp_eq(simd_vector a, simd_vector b); 

```

## logical

## misc

```C

// returns a sorted vector in ascending order
simd_vector simd_sort(simd_vector input);

// reverses the order of the vector
simd_vector simd_reverse(simd_vector a);

```
