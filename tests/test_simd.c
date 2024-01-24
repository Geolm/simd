#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include "greatest.h"
#include <math.h>
#include "../extra/vec2.h"
#include "../extra/simd_math.h"
#include "../extra/simd_approx_math.h"
#include "../extra/simd_color.h"

#include "test_simd_math.h"
#include "test_collision_2d.h"




TEST load_xy(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i;
    
    simd_vector x, y;
    simd_load_xy(array, &x, &y);
    
    simd_store(array, x);
    simd_store(array+simd_vector_width, y);
    
    for(int i=0; i<2; ++i)
        for(int j=0; j<simd_vector_width; ++j)
            ASSERT_EQ(array[i*simd_vector_width + j], (float)(i+(j*2)));

    PASS();
}

TEST load_xyz(void)
{
    float array[simd_vector_width*3];
    for(int i=0; i<simd_vector_width*3; ++i)
        array[i] = (float) i;

    simd_vector x, y, z;
    simd_load_xyz(array, &x, &y, &z);
    simd_store(array, x);
    simd_store(array+simd_vector_width, y);
    simd_store(array+simd_vector_width*2, z);
    
    for(int i=0; i<3; ++i)
        for(int j=0; j<simd_vector_width; ++j)
            ASSERT_EQ(array[i*simd_vector_width + j], (float)(i+(j*3)));

    PASS();
}

TEST load_xyzw(void)
{
    float array[simd_vector_width*4];
    for(int i=0; i<simd_vector_width*4; ++i)
        array[i] = (float) i;

    simd_vector x, y, z, w;
    simd_load_xyzw(array, &x, &y, &z, &w);
    simd_store(array, x);
    simd_store(array+simd_vector_width, y);
    simd_store(array+simd_vector_width*2, z);
    simd_store(array+simd_vector_width*3, w);
    
    for(int i=0; i<4; ++i)
        for(int j=0; j<simd_vector_width; ++j)
            ASSERT_EQ(array[i*simd_vector_width + j], (float)(i+(j*4)));

    PASS();
}

SUITE(load)
{
    RUN_TEST(load_xy);
    RUN_TEST(load_xyz);
    RUN_TEST(load_xyzw);
}

TEST sort(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (simd_vector_width-i);

    simd_vector a = simd_load(array);
    a = simd_sort(a);

    simd_store(array, a);
    
    for(int i=0; i<simd_vector_width-1; ++i)
        ASSERT_LT(array[i], array[i+1]);

    simd_store(array, simd_reverse(a));

    for(int i=0; i<simd_vector_width-1; ++i)
        ASSERT_GT(array[i], array[i+1]);

    PASS();
}

TEST get_lane(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (simd_vector_width-i);

    simd_vector a = simd_load(array);

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(simd_get_lane(a, i), array[i]);

    ASSERT_EQ(simd_get_first_lane(a), array[0]);
    PASS();
}

TEST hsum(void)
{
    float sum=0.0f;
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
    {
        array[i] = (float) i;
        sum += array[i];
    }

    simd_vector a = simd_load(array);
    ASSERT_EQ(simd_hsum(a), sum);

    PASS();
}

TEST hmin(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) i;
    
    simd_vector a = simd_load(array);
    ASSERT_EQ(simd_hmin(a), array[0]);
    PASS();
}

TEST hmax(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) i;
    
    simd_vector a = simd_load(array);
    ASSERT_EQ(simd_hmax(a), array[simd_last_lane]);
    PASS();
}

SUITE(horizontal)
{
    RUN_TEST(get_lane);
    RUN_TEST(hsum);
    RUN_TEST(hmin);
    RUN_TEST(hmax);
}

TEST fmad(void)
{
    float array[simd_vector_width*3];
    for(int i=0; i<simd_vector_width*3; ++i)
        array[i] = (float) i - (float)simd_vector_width;

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);
    simd_vector c = simd_load(array + simd_vector_width * 2);

    float result[simd_vector_width];
    simd_store(result, simd_fmad(a, b, c));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], (array[i] * array[i + simd_vector_width]) + array[i + simd_vector_width * 2]);

    PASS();
}

float addf(float x, float y) {return x+y;}
float subf(float x, float y) {return x-y;}
float mulf(float x, float y) {return x*y;}
float divf(float x, float y) {return x/y;}
float signf(float x) {if (x>=0) return 1; else return -1;}

SUITE(arithmetic)
{
    printf(".");
    RUN_TESTp(generic_test2, addf, simd_add, 0.f, false, "simd_add");
    RUN_TESTp(generic_test2, subf, simd_sub, 0.f, false, "simd_sub");
    RUN_TESTp(generic_test2, mulf, simd_mul, 0.f, false, "simd_mul");
    RUN_TESTp(generic_test, signf, simd_sign, -10.f, 10.f, 0.f, false, "simd_sign");
    
    RUN_TEST(fmad);
}

float rcp(float x) {return 1.f/x;}
float rsqrt(float x) {return 1.f/sqrtf(x);}

SUITE(sqrt_and_rcp)
{
    printf(".");
    RUN_TESTp(generic_test, rcp, simd_rcp, 0.1f, 100.f, 1e-3f, true, "simd_rcp");
    RUN_TESTp(generic_test, rsqrt, simd_rsqrt, 0.1f, 100.f, 1e-3f, true, "simd_rsqrt");
    RUN_TESTp(generic_test, sqrtf, simd_sqrt, 0.1f, 100.f, 0.f, true, "simd_sqrt");
}

float negative(float x) {return -x;}

SUITE(abs_neg_min_max)
{
    printf(".");
    RUN_TESTp(generic_test, fabsf, simd_abs, -10.f, 10.f, 0.f, false, "simd_abs");
    RUN_TESTp(generic_test, negative, simd_neg, -10.f, 10.f, 0.f, false, "simd_neg");
    RUN_TESTp(generic_test2, float_min, simd_min, 0.f, false, "simd_min");
    RUN_TESTp(generic_test2, float_max, simd_max, 0.f, false, "simd_max");
}

TEST export_int16(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (i-32767);

    simd_vector a = simd_load(array);

    int16_t output[simd_vector_width];
    simd_export_int16(a, output);

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(i-32767, output[i]);

    PASS();
}

TEST export_int8(void)
{
    float array[simd_vector_width*4];
    for(int i=0; i<simd_vector_width*4; ++i)
        array[i] = (float) (i-128);

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array+simd_vector_width);
    simd_vector c = simd_load(array+simd_vector_width*2);
    simd_vector d = simd_load(array+simd_vector_width*3);

    int8_t output[simd_vector_width*4];
    simd_export_int8(a, b, c, d, output);

    for(int i=0; i<simd_vector_width*4; ++i)
        ASSERT_EQ(i-128, output[i]);

    PASS();
}

TEST export_uint8(void)
{
    float array[simd_vector_width*4];
    for(int i=0; i<simd_vector_width*4; ++i)
        array[i] = (float) (255-i);

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array+simd_vector_width);
    simd_vector c = simd_load(array+simd_vector_width*2);
    simd_vector d = simd_load(array+simd_vector_width*3);

    uint8_t output[simd_vector_width*4];
    simd_export_uint8(a, b, c, d, output);

    for(int i=0; i<simd_vector_width*4; ++i)
        ASSERT_EQ(255-i, output[i]);

    PASS();
}

SUITE(export)
{
    RUN_TEST(export_int16);
    RUN_TEST(export_int8);
    RUN_TEST(export_uint8);
}

TEST interlace_xy(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) (i);

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);

    simd_vector interlace0, interlace1;
    simd_interlace_xy(a, b, &interlace0, &interlace1);

    float result[simd_vector_width*2];
    simd_store(result, interlace0);
    simd_store(result+simd_vector_width, interlace1);

    for(int i=0; i<simd_vector_width; ++i)
    {
        ASSERT_EQ(result[i*2], array[i]);
        ASSERT_EQ(result[i*2+1], array[i+simd_vector_width]);
    }

    PASS();
}

TEST interlace_xyzw(void)
{
    float array[simd_vector_width*4];
    for(int i=0; i<simd_vector_width*4; ++i)
        array[i] = (float) (i);

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);
    simd_vector c = simd_load(array + simd_vector_width * 2);
    simd_vector d = simd_load(array + simd_vector_width * 3);

    simd_vector interlace0, interlace1, interlace2, interlace3;
    simd_interlace_xyzw(a, b, c, d, &interlace0, &interlace1, &interlace2, &interlace3);

    float result[simd_vector_width*4];
    simd_store(result, interlace0);
    simd_store(result+simd_vector_width, interlace1);
    simd_store(result+simd_vector_width*2, interlace2);
    simd_store(result+simd_vector_width*3, interlace3);

    for(int i=0; i<simd_vector_width; ++i)
    {
        ASSERT_EQ(result[i*4], array[i]);
        ASSERT_EQ(result[i*4+1], array[i+simd_vector_width]);
        ASSERT_EQ(result[i*4+2], array[i+simd_vector_width*2]);
        ASSERT_EQ(result[i*4+3], array[i+simd_vector_width*3]);
    }

    PASS();
}

SUITE(interlace_deinterlace)
{
    RUN_TEST(interlace_xy);
    RUN_TEST(interlace_xyzw);
}

float fractf(float x) {return x - (int) x;}

SUITE(rounding)
{
    printf(".");
    RUN_TESTp(generic_test, floorf, simd_floor, -100.0f, 100.f, 0.0f, false, "simd_floor");
    RUN_TESTp(generic_test, ceilf, simd_ceil, -100.0f, 100.f, 0.0f, false, "simd_ceil");
    RUN_TESTp(generic_test, fractf, simd_fract, -100.0f, 100.f, 0.0f, false, "simd_fract");
}

TEST test_frexp(void)
{
    for(int j=0; j<1000; ++j)
    {
        float array[simd_vector_width];
        float mantissa[simd_vector_width];
        int exponent[simd_vector_width];
        for(int i=0; i<simd_vector_width; ++i)
        {
            array[i] = (float) (i + j - 500);
            mantissa[i] = frexpf(array[i], &exponent[i]);
        }

        simd_vector v_input = simd_load(array);
        simd_vector v_exponent;
        simd_vector v_mantissa = simd_frexp(v_input, &v_exponent);

        float exponent_export[simd_vector_width];
        simd_store(exponent_export, v_exponent);

        for(int i=0; i<simd_vector_width; ++i)
            ASSERT_EQ((int)exponent_export[i], exponent[i]);

        ASSERT( simd_all(simd_cmp_eq(v_mantissa, simd_load(mantissa))));
    }
    PASS();
}

TEST test_ldexp(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (simd_vector_width/2-i);

    float result[simd_vector_width];
    float param[simd_vector_width];

    simd_vector v_array = simd_load(array);

    for(int i=-127; i<(128-simd_vector_width); i+= simd_vector_width)
    {
        for(int j=0; j<simd_vector_width; ++j)
        {
            result[j] = ldexpf(array[j], i + j);
            param[j] = (float)(i + j); 
        }

        simd_vector v_result = simd_ldexp(v_array, simd_load(param));
        
        ASSERT(simd_all(simd_cmp_eq(simd_load(result), v_result)));
    }

    PASS();
}

SUITE(float_extraction)
{
    RUN_TEST(test_frexp);
    RUN_TEST(test_ldexp);
}

GREATEST_MAIN_DEFS();

int main(int argc, char * argv[])
{
    GREATEST_MAIN_BEGIN();

    RUN_SUITE(load);
    RUN_SUITE(rounding);
    RUN_TEST(sort);
    RUN_SUITE(float_extraction);
    RUN_SUITE(horizontal);
    RUN_SUITE(trigonometry);
    RUN_SUITE(exponentiation);
    RUN_SUITE(color_space);
    RUN_SUITE(arithmetic);
    RUN_SUITE(sqrt_and_rcp);
    RUN_SUITE(abs_neg_min_max);
    RUN_SUITE(export);
    RUN_SUITE(collision_2d);
    RUN_SUITE(interlace_deinterlace);

    GREATEST_MAIN_END();
}
