# simd
Neon/AVX simd library, vector size agnostic


# documentation

The idea of the libray is to not assume a specific simd vector width (4 for SSE/Neon, 8 for AVX and so on) but use **simd_vector_width** variable instead. As a result the library does not contains function to get lanes value, nor do shuffle based on the width of the vector. The API is based on AVX and Neon, some functions are composed of multiple simd instructions.


simd_vector is typedef to the native simd vector of the platform (avx or neon).


## load/store functions

* simd_vector simd_load(const float* array) : returns a vector loaded for the pointer [**array**] of float
* simd_vector simd_load_partial(const float* array, int count, float unload_value) : load a partial array of [**count**] floats, fill the rest with [**unload_value**]
* void simd_store(float* array, simd_vector a) : store a vector [**a**] at the pointer [**array**] 



## arithmetic functions
* simd_add(a, b) : returns a simd_vector a + b
* 

## logical functions

## other functions
* simd_sort : sort lanes
* 
