#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include "greatest.h"
#include <math.h>
#include "../vec2.h"
#include "../simd_math.h"

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

TEST sign(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i - (float)simd_vector_width;
    
    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);
    
    float result[simd_vector_width*2];
    
    simd_store(result, simd_sign(a));
    simd_store(result+simd_vector_width, simd_sign(b));
    
    for(int i=0; i<simd_vector_width*2; ++i)
        ASSERT_EQ(result[i], float_sign(array[i]));

    PASS();
}

TEST add(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i - (float)simd_vector_width;

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);

    float result[simd_vector_width];
    simd_store(result, simd_add(a, b));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], array[i] + array[i + simd_vector_width]);

    PASS();
}

TEST sub(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i - (float)simd_vector_width;

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);

    float result[simd_vector_width];
    simd_store(result, simd_sub(a, b));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], array[i] - array[i + simd_vector_width]);

    PASS();
}

TEST mul(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i - (float)simd_vector_width;

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);

    float result[simd_vector_width];
    simd_store(result, simd_mul(a, b));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], array[i] * array[i + simd_vector_width]);

    PASS();
}

TEST division(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i - (float)simd_vector_width;

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);

    float result[simd_vector_width];
    simd_store(result, simd_div(a, b));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], array[i] / array[i + simd_vector_width]);

    PASS();
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

SUITE(arithmetic)
{
    RUN_TEST(sign);
    RUN_TEST(add);
    RUN_TEST(sub);
    RUN_TEST(mul);
    RUN_TEST(division);
    RUN_TEST(fmad);
}

TEST squareroot(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) i;

    simd_vector a = simd_load(array);

    float result[simd_vector_width];
    simd_store(result, simd_sqrt(a));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], sqrtf(array[i]));

    PASS();
}

TEST rcp_squareroot(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (i + 1);

    simd_vector a = simd_load(array);

    float result[simd_vector_width];
    simd_store(result, simd_rsqrt(a));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_LT(fabsf(result[i] - (1.f / sqrtf(array[i]))), 0.0005f);

    PASS();
}

TEST rcp(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (i + 1);

    simd_vector a = simd_load(array);

    float result[simd_vector_width];
    simd_store(result, simd_rcp(a));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_LT(fabsf(result[i] - (1.f / array[i])), 0.0005f);

    PASS();
}

SUITE(sqrt_and_rcp)
{
    RUN_TEST(squareroot);
    RUN_TEST(rcp_squareroot);
    RUN_TEST(rcp);
}

TEST absolute(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (-i);
    
    float result[simd_vector_width];
    simd_store(result, simd_abs(simd_load(array)));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], fabsf(array[i]));

    PASS();
}

TEST negative(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (i) - (simd_vector_width/2);
    
    float result[simd_vector_width];
    simd_store(result, simd_neg(simd_load(array)));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], -array[i]);

    PASS();
}

TEST minimum(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i - (float)simd_vector_width;

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);

    float result[simd_vector_width];
    simd_store(result, simd_min(a, b));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], float_min(array[i], array[i+simd_vector_width]));

    PASS();
}

TEST maximum(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i - (float)simd_vector_width;

    simd_vector a = simd_load(array);
    simd_vector b = simd_load(array + simd_vector_width);

    float result[simd_vector_width];
    simd_store(result, simd_max(a, b));

    for(int i=0; i<simd_vector_width; ++i)
        ASSERT_EQ(result[i], float_max(array[i], array[i+simd_vector_width]));

    PASS();
}

SUITE(abs_neg_min_max)
{
    RUN_TEST(absolute);
    RUN_TEST(negative);
    RUN_TEST(minimum);
    RUN_TEST(maximum);
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

GREATEST_MAIN_DEFS();

int main(int argc, char * argv[])
{
    GREATEST_MAIN_BEGIN();

    RUN_SUITE(load);
    RUN_TEST(sort);
    RUN_SUITE(horizontal);
    RUN_SUITE(trigonometry);
    RUN_SUITE(arithmetic);
    RUN_SUITE(sqrt_and_rcp);
    RUN_SUITE(abs_neg_min_max);
    RUN_TEST(approx_length);
    RUN_SUITE(export);
    RUN_SUITE(collision_2d);
    RUN_SUITE(interlace_deinterlace);

    GREATEST_MAIN_END();
}
