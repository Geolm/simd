# simd_math

Provides transcendental functions that are missing in Neon/AVX instructions.

Heavily based on cephes implementation of C math functions, high precision.

# functions

```C

// max error : 5.960464478e-08
simd_vector simd_sin(simd_vector x);

// max error : 5.960464478e-08
simd_vector simd_cos(simd_vector x);

// max error : 5.960464478e-08
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c);

// max error : 2.384185791e-07
simd_vector simd_acos(simd_vector x);

// max error : 1.192092896e-07
simd_vector simd_asin(simd_vector x);

// max error : 6.699562073e-05
simd_vector simd_atan(simd_vector x);

// max error : 2.384185791e-07
simd_vector simd_atan2(simd_vector x, simd_vector y);

// max error : 4.768371582e-07
simd_vector simd_log(simd_vector x);

// max error : 1.108270880e-07
simd_vector simd_exp(simd_vector x);
```

# simd_approx_math

The simd_approx_math.h header provides faster math functions with less precision.

```C
// max error : 1.251697540e-06, ~2.5x faster than simd_cos
simd_vector simd_approx_cos(simd_vector a);

// max error : 1.192092896e-06, ~2.5x faster than simd_cos
simd_vector simd_approx_sin(simd_vector a);

// max error : 6.520748138e-05, ~2.8x faster than simd_acos
simd_vector simd_approx_acos(simd_vector x);

// max relative error : 1.727835275e-03, ~3.7x faster than simd_exp
simd_vector simd_approx_exp(simd_vector x);

// max error : 4.267692566e-05
simd_vector simd_approx_srgb_to_linear(simd_vector value);

// max error : 3.309000051e-03 (enough for 8bits value)
simd_vector simd_approx_linear_to_srgb(simd_vector value); 
```
