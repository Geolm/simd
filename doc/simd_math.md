# simd_math

Provides transcendental functions that are missing in Neon/AVX instructions.

Heavily based on cephes implementation of C math functions, if not specified output is identical to C functions (see below)

# trigonometric functions

```C
simd_vector simd_sin(simd_vector x);
simd_vector simd_cos(simd_vector x);
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c);
simd_vector simd_acos(simd_vector x); 
simd_vector simd_asin(simd_vector x); 
```

```C
// max error : 0.000002
simd_vector simd_atan2(simd_vector x, simd_vector y); 
```
