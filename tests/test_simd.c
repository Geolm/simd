#include <stdio.h>
#include "greatest.h"
#include <math.h>
#include "../simd_math.h"

#include "test_simd_math.h"



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


static inline float float_sign(float f) {if (f>0.f) return 1.f; if (f<0.f) return -1.f; return 0.f;}

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

SUITE(arithmetic)
{
    RUN_TEST(sign);
    RUN_TEST(add);
    RUN_TEST(sub);
    RUN_TEST(mul);
    RUN_TEST(division);
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

    GREATEST_MAIN_END();
}
