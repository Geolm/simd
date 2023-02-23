# simd_collision

This library uses simd.h to compute intersection between 2d primitives with AVX/Neon instructions under the hood. In order to maximize the size of the simd register, the library batches intersection requests and so the results are deferred.


