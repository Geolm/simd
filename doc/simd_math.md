# simd_math

Provides transcendental functions that are missing in Neon/AVX instructions.

Heavily based on cephes implementation of C math functions, if not specified output is identical to C functions (see below)

# functions

```C
simd_vector simd_sin(simd_vector x);
simd_vector simd_cos(simd_vector x);
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c);
simd_vector simd_acos(simd_vector x); 
simd_vector simd_asin(simd_vector x);
simd_vector simd_log(simd_vector x);
simd_vector simd_exp(simd_vector x);
```

```C
// max error : 2.38418579E-7
simd_vector simd_atan2(simd_vector x, simd_vector y);

// max error with input [-10; 10]: 0.000002
simd_vector simd_atan(simd_vector x); 
```
