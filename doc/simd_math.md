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

// max error : 4.768371582e-07
simd_vector simd_cbrt(simd_vector x);
```

# simd_approx_math

The simd_approx_math.h header provides faster math functions with less precision.

```C
// max error : 5.811452866e-07, ~2.5x faster than simd_cos
simd_vector simd_approx_cos(simd_vector a); 

// max error : 2.682209015e-07, ~2.5x faster than simd_cos
simd_vector simd_approx_sin(simd_vector a); 

// max error : 0.000068, ~2.8x faster than simd_acos
simd_vector simd_approx_acos(simd_vector x);

// max relative error : 0.001726, ~3.7x faster than simd_exp
simd_vector simd_approx_exp(simd_vector x); 
```
