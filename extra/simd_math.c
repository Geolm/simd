#include "simd_math.h"


//----------------------------------------------------------------------------------------------------------------------
// based on https://github.com/jeremybarnes/cephes/blob/master/single/atanf.c
simd_vector simd_atan(simd_vector xx)
{	
	simd_vector sign = simd_sign(xx);
	simd_vector x = simd_abs(xx);
	simd_vector one = simd_splat(1.f);

	// range reduction
	simd_vector above_3pi8 = simd_cmp_gt(x, simd_splat(2.414213562373095f));
	simd_vector above_pi8 = simd_andnot(simd_cmp_gt(x, simd_splat(0.4142135623730950f)), above_3pi8);
	simd_vector y = simd_splat_zero();
	x = simd_select(x, simd_neg(simd_rcp(x)), above_3pi8);
	x = simd_select(x, simd_div(simd_sub(x, one), simd_add(x, one)), above_pi8);
	y = simd_select(y, simd_splat(SIMD_MATH_PI2), above_3pi8);
	y = simd_select(y, simd_splat(SIMD_MATH_PI4), above_pi8);
	
	// minimax polynomial
	simd_vector z = simd_mul(x, x);
	simd_vector tmp = simd_fmad(z, simd_splat(8.05374449538e-2f), simd_splat(-1.38776856032E-1f));
	tmp = simd_fmad(tmp, z, simd_splat(1.99777106478E-1f));
	tmp = simd_fmad(tmp, z, simd_splat(-3.33329491539E-1f));
    tmp = simd_mul(tmp, z);
	tmp = simd_fmad(tmp, x, x);
	y = simd_add(tmp, y);
	y = simd_mul(y, sign);
	
	return y;	
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://mazzo.li/posts/vectorized-atan2.html
simd_vector simd_atan2(simd_vector x, simd_vector y)
{
    simd_vector swap = simd_cmp_lt(simd_abs(x), simd_abs(y));
    simd_vector x_equals_zero = simd_cmp_eq(x, simd_splat_zero());
    simd_vector y_equals_zero = simd_cmp_eq(y, simd_splat_zero());
    simd_vector x_over_y = simd_div(x, y);
    simd_vector y_over_x = simd_div(y, x);
    simd_vector atan_input = simd_select(y_over_x, x_over_y, swap);
    simd_vector result = simd_atan(atan_input);

    simd_vector adjust = simd_select(simd_splat(-SIMD_MATH_PI2), simd_splat(SIMD_MATH_PI2), simd_cmp_ge(atan_input, simd_splat_zero()));
    result = simd_select(result, simd_sub(adjust, result), swap);

    simd_vector x_sign_mask = simd_cmp_lt(x, simd_splat_zero());
    result = simd_add( simd_and(simd_xor(simd_splat(SIMD_MATH_PI), simd_and(simd_sign_mask(), y)), x_sign_mask), result);
    result = simd_select(result, simd_mul(simd_sign(x), simd_splat_zero()), y_equals_zero);
    result = simd_select(result, simd_mul(simd_sign(y), simd_splat(SIMD_MATH_PI2)), x_equals_zero);
    return result;
}

//----------------------------------------------------------------------------------------------------------------------
// based on http://gruntthepeon.free.fr/ssemath/
simd_vector simd_log(simd_vector x)
{
    simd_vector one = simd_splat(1.f);
    simd_vector invalid_mask = simd_cmp_le(x, simd_splat_zero());
    x = simd_max(x, simd_min_normalized());  // cut off denormalized stuff

    simd_vectori emm0 = simd_shift_right_i(simd_cast_from_float(x), 23);
    emm0 = simd_sub_i(emm0, simd_splat_i(0x7f));
    simd_vector e = simd_convert_from_int(emm0);
    
    // keep only the fractional part
    x = simd_and(x, simd_inv_mant_mask());
    x = simd_or(x, simd_splat(0.5f));
    
    e = simd_add(e, one);
    simd_vector mask = simd_cmp_lt(x, simd_splat(0.707106781186547524f));
    simd_vector tmp = simd_and(x, mask);
    x = simd_sub(x, one);
    e = simd_sub(e, simd_and(one, mask));
    x = simd_add(x, tmp);

    simd_vector z = simd_mul(x,x);
    simd_vector y = simd_splat(7.0376836292E-2f);
    y = simd_fmad(y, x, simd_splat(-1.1514610310E-1f));
    y = simd_fmad(y, x, simd_splat(1.1676998740E-1f));
    y = simd_fmad(y, x, simd_splat(-1.2420140846E-1f));
    y = simd_fmad(y, x, simd_splat(+1.4249322787E-1f));
    y = simd_fmad(y, x, simd_splat(-1.6668057665E-1f));
    y = simd_fmad(y, x, simd_splat(+2.0000714765E-1f));
    y = simd_fmad(y, x, simd_splat(-2.4999993993E-1f));
    y = simd_fmad(y, x, simd_splat(+3.3333331174E-1f));
    y = simd_mul(y, x);
    y = simd_mul(y, z);

    tmp = simd_mul(e, simd_splat(-2.12194440e-4f));
    y = simd_add(y, tmp);

    tmp = simd_mul(z, simd_splat(0.5f));
    y = simd_sub(y, tmp);

    tmp = simd_mul(e, simd_splat(0.693359375f));
    x = simd_add(x, y);
    x = simd_add(x, tmp);
    x = simd_or(x, invalid_mask); // negative arg will be NAN

    return x;
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://github.com/redorav/hlslpp/blob/master/include/hlsl%2B%2B_vector_float8.h
simd_vector simd_log2(simd_vector x)
{
    simd_vector one = simd_splat(1.f);
    simd_vectori exp = simd_splat_i(0x7f800000);
    simd_vectori mant = simd_splat_i(0x007fffff);
    simd_vectori i = simd_cast_from_float(x);
    simd_vector e = simd_convert_from_int(simd_sub_i(simd_shift_right_i(simd_and_i(i, exp), 23), simd_splat_i(127)));
    simd_vector m = simd_or(simd_cast_from_int(simd_and_i(i, mant)), one);

    // minimax polynomial fit of log2(x)/(x - 1), for x in range [1, 2[
    simd_vector p = simd_fmad(m, simd_splat(-3.4436006e-2f), simd_splat(3.1821337e-1f));
    p = simd_fmad(m, p, simd_splat(-1.2315303f));
    p = simd_fmad(m, p, simd_splat(2.5988452f));
    p = simd_fmad(m, p, simd_splat(-3.3241990f));
    p = simd_fmad(m, p, simd_splat(3.1157899f));

    // this effectively increases the polynomial degree by one, but ensures that log2(1) == 0
    p = simd_mul(p, simd_sub(m, one));
    simd_vector result = simd_add(p, e);

    // we can't compute a logarithm beyond this value, so we'll mark it as -infinity to indicate close to 0
    simd_vector ltminus127 = simd_cmp_le(result, simd_splat(-127.f));
    result = simd_select(result, simd_splat_negative_infinity(), ltminus127);

    // Check for negative values and return NaN
    simd_vector lt0 = simd_cmp_lt(x, simd_splat_zero());
    result = simd_select(result, simd_splat_nan(), lt0);

    return result;
}

//----------------------------------------------------------------------------------------------------------------------
// based on http://gruntthepeon.free.fr/ssemath/
simd_vector simd_exp(simd_vector x)
{
    simd_vector tmp = simd_splat_zero();
    simd_vector fx;
    simd_vector one = simd_splat(1.f);

    x = simd_min(x, simd_splat(88.3762626647949f));
    x = simd_max(x, simd_splat(-88.3762626647949f));

    // express exp(x) as exp(g + n*log(2))
    fx = simd_fmad(x, simd_splat(1.44269504088896341f), simd_splat(0.5f));
    tmp = simd_floor(fx);

    // if greater, substract 1
    simd_vector mask = simd_cmp_gt(tmp, fx);
    mask = simd_and(mask, one);
    fx = simd_sub(tmp, mask);

    tmp = simd_mul(fx, simd_splat(0.693359375f));
    simd_vector z = simd_mul(fx, simd_splat(-2.12194440e-4f));
    x = simd_sub(x, tmp);
    x = simd_sub(x, z);
    z = simd_mul(x, x);
    simd_vector y = simd_splat(1.9875691500E-4f);
    y = simd_fmad(y, x, simd_splat(1.3981999507E-3f));
    y = simd_fmad(y, x, simd_splat(8.3334519073E-3f));
    y = simd_fmad(y, x, simd_splat(4.1665795894E-2f));
    y = simd_fmad(y, x, simd_splat(1.6666665459E-1f));
    y = simd_fmad(y, x, simd_splat(5.0000001201E-1f));
    y = simd_fmad(y, z, x);
    y = simd_add(y, one);

    simd_vectori emm0 = simd_convert_from_float(fx);
    emm0 = simd_add_i(emm0, simd_splat_i(0x7f));
    emm0 = simd_shift_left_i(emm0, 23);
    simd_vector pow2n = simd_cast_from_int(emm0);

    y = simd_mul(y, pow2n);
    return y;
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://github.com/jeremybarnes/cephes/blob/master/single/exp2f.c
simd_vector simd_exp2(simd_vector x)
{
    // clamp values
    x = simd_clamp(x, simd_splat(-127.f), simd_splat(127.f));
    simd_vector equal_to_zero = simd_cmp_eq(x, simd_splat_zero());

    simd_vector i0 = simd_floor(x);
    x = simd_sub(x, i0);

    simd_vector above_half = simd_cmp_gt(x, simd_splat(.5f));
    simd_vector one = simd_splat(1.f);
    i0 = simd_select(i0, simd_add(i0, one), above_half);
    x = simd_select(x, simd_sub(x, one), above_half);

    simd_vector px = simd_fmad(x, simd_splat(1.535336188319500E-004f), simd_splat(1.339887440266574E-003f));
    px = simd_fmad(px, x, simd_splat(9.618437357674640E-003f));
    px = simd_fmad(px, x, simd_splat(5.550332471162809E-002f));
    px = simd_fmad(px, x, simd_splat(2.402264791363012E-001f));
    px = simd_fmad(px, x, simd_splat(6.931472028550421E-001f));
    px = simd_fmad(px, x,  one);
    px = simd_ldexp(px, i0);

    return simd_select(px, one, equal_to_zero);
}

//----------------------------------------------------------------------------------------------------------------------
// based on http://gruntthepeon.free.fr/ssemath/
void simd_sincos(simd_vector x, simd_vector* s, simd_vector* c)
{
    simd_vector xmm1, xmm2, xmm3 = simd_splat_zero(), sign_bit_sin, y;

    sign_bit_sin = x;

    // take the absolute value
    x = simd_and(x, simd_inv_sign_mask());
    // extract the sign bit (upper one)
    sign_bit_sin = simd_and(sign_bit_sin, simd_sign_mask());

    // scale by 4/Pi
    y = simd_mul(x, simd_splat(1.27323954473516f));

    // store the integer part of y in emm2 
    simd_vectori emm2 = simd_convert_from_float(y);

    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = simd_add_i(emm2, simd_splat_i(1));
    emm2 = simd_and_i(emm2, simd_splat_i(~1));
    y = simd_convert_from_int(emm2);

    simd_vectori emm4 = emm2;

    // get the swap sign flag for the sine
    simd_vectori emm0 = simd_and_i(emm2, simd_splat_i(4));
    emm0 = simd_shift_left_i(emm0, 29);
    simd_vector swap_sign_bit_sin = simd_cast_from_int(emm0);

    // get the polynom selection mask for the sine
    emm2 = simd_and_i(emm2, simd_splat_i(2));
    emm2 = simd_cmp_eq_i(emm2, simd_splat_zero_i());
    simd_vector poly_mask = simd_cast_from_int(emm2); 

    // The magic pass: "Extended precision modular arithmetic" 
    //  x = ((x - y * DP1) - y * DP2) - y * DP3; 
    x = simd_fmad(y, simd_splat(-0.78515625f), x);
    x = simd_fmad(y, simd_splat(-2.4187564849853515625e-4f), x);
    x = simd_fmad(y, simd_splat(-3.77489497744594108e-8f), x);

    emm4 = simd_sub_i(emm4, simd_splat_i(2));
    emm4 = simd_andnot_i(simd_splat_i(4), emm4);
    emm4 = simd_shift_left_i(emm4, 29);
    simd_vector sign_bit_cos = simd_cast_from_int(emm4); 

    sign_bit_sin = simd_xor(sign_bit_sin, swap_sign_bit_sin);
    
    // Evaluate the first polynom  (0 <= x <= Pi/4)
    simd_vector z = simd_mul(x,x);
    y = simd_splat(2.443315711809948E-005f);
    y = simd_fmad(y, z, simd_splat(-1.388731625493765E-003f));
    y = simd_fmad(y, z, simd_splat(4.166664568298827E-002f));
    y = simd_mul(y, z);
    y = simd_mul(y, z);
    simd_vector tmp = simd_mul(z, simd_splat(.5f));
    y = simd_sub(y, tmp);
    y = simd_add(y, simd_splat(1.f));

    // Evaluate the second polynom  (Pi/4 <= x <= 0)
    simd_vector y2 = simd_splat(-1.9515295891E-4f);
    y2 = simd_fmad(y2, z, simd_splat(8.3321608736E-3f));
    y2 = simd_fmad(y2, z, simd_splat(-1.6666654611E-1f));
    y2 = simd_mul(y2, z);
    y2 = simd_mul(y2, x);
    y2 = simd_add(y2, x);

    // select the correct result from the two polynoms
    xmm3 = poly_mask;
    simd_vector ysin2 = simd_and(y2, xmm3);
    simd_vector ysin1 = simd_andnot(y, xmm3);
    y2 = simd_sub(y2,ysin2);
    y = simd_sub(y, ysin1);

    xmm1 = simd_add(ysin1,ysin2);
    xmm2 = simd_add(y,y2);

    // update the sign
    *s = simd_xor(xmm1, sign_bit_sin);
    *c = simd_xor(xmm2, sign_bit_cos);
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://github.com/jeremybarnes/cephes/blob/master/single/asinf.c
simd_vector simd_asin(simd_vector xx)
{
    simd_vector x = xx;
    simd_vector sign = simd_sign(xx);
    simd_vector a  = simd_abs(x);
    simd_vector greater_one = simd_cmp_gt(a, simd_splat(1.f));
    simd_vector small_value = simd_cmp_lt(a, simd_splat(1.0e-4f));

    simd_vector z1 = simd_mul(simd_splat(.5f), simd_sub(simd_splat(1.f), a));
    simd_vector z2 = simd_mul(a, a);
    simd_vector flag = simd_cmp_gt(a, simd_splat(.5f));
    simd_vector z = simd_select(z2, z1, flag);

    x = simd_select(a, simd_sqrt(z), flag);

    simd_vector tmp = simd_fmad(simd_splat(4.2163199048E-2f), z, simd_splat(2.4181311049E-2f));
    tmp = simd_fmad(tmp, z, simd_splat(4.5470025998E-2f));
    tmp = simd_fmad(tmp, z, simd_splat(7.4953002686E-2f));
    tmp = simd_fmad(tmp, z, simd_splat(1.6666752422E-1f));
    tmp = simd_mul(tmp, z);
    z = simd_fmad(tmp, x, x);

    tmp = simd_add(z, z);
    tmp = simd_sub(simd_splat(SIMD_MATH_PI2), tmp);
    z = simd_select(z, tmp, flag);

    z = simd_select(z, a, small_value);
    z = simd_select(z, simd_splat_zero(), greater_one);
    z = simd_mul(z, sign);
    return z;
}

//----------------------------------------------------------------------------------------------------------------------
// acos(x) = pi/2 - asin(x)
simd_vector simd_acos(simd_vector x)
{
    simd_vector out_of_bound = simd_cmp_gt(simd_abs(x), simd_splat(1.f));
    simd_vector result = simd_sub(simd_splat(SIMD_MATH_PI2), simd_asin(x));
    result = simd_select(result, simd_splat_zero(), out_of_bound);
    return result;
}

//----------------------------------------------------------------------------------------------------------------------
// based on https://github.com/jeremybarnes/cephes/blob/master/single/cbrtf.c
simd_vector simd_cbrt(simd_vector xx)
{
    simd_vector one_over_three = simd_splat(0.333333333333f);
    simd_vector sign = simd_sign(xx);
    simd_vector x = simd_abs(xx);
    simd_vector z = x;

    // extract power of 2, leaving mantissa between 0.5 and 1
    simd_vector exponent;
    x = simd_frexp(x, &exponent);

    // Approximate cube root of number between .5 and 1
    simd_vector tmp = simd_fmad(simd_splat(-0.13466110473359520655053f), x, simd_splat(0.54664601366395524503440f));
    tmp = simd_fmad(tmp, x, simd_splat(-0.95438224771509446525043f));
    tmp = simd_fmad(tmp, x, simd_splat(1.1399983354717293273738f));
    x = simd_fmad(tmp, x, simd_splat(0.40238979564544752126924f));

    // exponent divided by 3
    simd_vector exponent_is_negative = simd_cmp_lt(exponent, simd_splat_zero());
    
    exponent = simd_abs(exponent);
    simd_vector rem = exponent;
    exponent = simd_floor(simd_mul(exponent, one_over_three));
    rem = simd_sub(rem, simd_mul(exponent, simd_splat(3.f)));

    simd_vector cbrt2 = simd_splat(1.25992104989487316477f);
    simd_vector cbrt4 = simd_splat(1.58740105196819947475f);

    simd_vector rem_equals_1 = simd_cmp_eq(rem, simd_splat(1.f));
    simd_vector rem_equals_2 = simd_cmp_eq(rem, simd_splat(2.f));
    simd_vector x1 = simd_mul(x, simd_select(cbrt4, cbrt2, rem_equals_1));
    simd_vector x2 = simd_div(x, simd_select(cbrt4, cbrt2, rem_equals_1));
	x = simd_select(x, simd_select(x1, x2, exponent_is_negative), simd_or(rem_equals_1, rem_equals_2));
    exponent = simd_mul(exponent, simd_select(simd_splat(1.f), simd_splat(-1.f), exponent_is_negative));

    // multiply by power of 2
    x = simd_ldexp(x, exponent);

    // Newton iteration, x -= ( x - (z/(x*x)) ) * 0.333333333333;
    x = simd_sub(x, simd_mul(simd_sub(x, simd_div(z, simd_mul(x, x))), one_over_three));
    x = simd_mul(x, sign);  // if input is zero, sign is also zero

    return x;
}



