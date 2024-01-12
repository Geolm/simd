# simd_math

Provides transcendental functions that are missing in Neon/AVX instructions.

Heavily based on cephes implementation of C math functions, if not specified output is identical to C functions (see below)

# functions

```C

// same output as C math lib sinf()
simd_vector simd_sin(simd_vector x);

// same output as C math lib cosf()
simd_vector simd_cos(simd_vector x);

// same output as C math lib sinf()/cosf()
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c);

// same output as C math lib acosf()
simd_vector simd_acos(simd_vector x);

// same output as C math lib asinf() 
simd_vector simd_asin(simd_vector x);

// same output as C math lib logf()
simd_vector simd_log(simd_vector x);

// same output as C math lib expf()
simd_vector simd_exp(simd_vector x);
```

```C
// max error : 2.38418579E-7
simd_vector simd_atan2(simd_vector x, simd_vector y);

// max error with input [-10; 10]: 0.000002
simd_vector simd_atan(simd_vector x); 
```

# simd_approx_math

Sometime you're more interested in speed than precision. The simd_approx_math.h header provides faster math functions with less precision.

```C
// max error : 0.000001, ~2.5x faster than simd_cos
simd_vector simd_approx_cos(simd_vector a);

// max error : 0.000001, ~2.5x faster than simd_cos
simd_vector simd_approx_sin(simd_vector a);

// max error : 0.000068, ~2.8x faster than simd_acos
simd_vector simd_approx_acos(simd_vector x);

// max relative error : 0.001726, ~3.7x faster than simd_cos
simd_vector simd_approx_exp(simd_vector x);

// max error : 0.000043
simd_vector simd_approx_srgb_to_linear(simd_vector value);

// max error : 0.003322 (enough for 8bits value)
simd_vector simd_approx_linear_to_srgb(simd_vector value); 
```
