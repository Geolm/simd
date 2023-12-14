
#include "../vec2.h"

#define NUM_VECTORS (100)
#define NUM_ELEMENTS (simd_vector_width * NUM_VECTORS)

TEST sinus(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = VEC2_TAU / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - VEC2_PI;
        result[i] = sinf(array[i]);
    }

    simd_vector epsilon = simd_splat(0.000002f);

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_sin(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));
    }

    PASS();
}

TEST approx_sin(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = VEC2_TAU / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - VEC2_PI;
        result[i] = sinf(array[i]);
    }

    simd_vector epsilon = simd_splat(0.005f);

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_approx_sin(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));
    }

    PASS();
}

TEST arcos(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 2.f / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - 1.f;
        result[i] = acosf(array[i]);
    }

    simd_vector epsilon = simd_splat(0.0001f);

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_acos(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));
    }

    PASS();
}

TEST approx_arcos(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 2.f / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - 1.f;
        result[i] = acosf(array[i]);
    }

    simd_vector epsilon = simd_splat(0.02f);

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_approx_acos(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));
    }

    PASS();
}

TEST arctan(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 2.f / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - 1.f;
        result[i] = atanf(array[i]);
    }

    simd_vector epsilon = simd_splat(0.00001f);

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_approx_atan(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));
    }

    PASS();
}

SUITE(trigonometry)
{
    RUN_TEST(sinus);
    RUN_TEST(approx_sin);
    RUN_TEST(arcos);
    RUN_TEST(approx_arcos);
    RUN_TEST(arctan);
}

TEST approx_length(void)
{
    vec2 array[NUM_ELEMENTS];
    float step = VEC2_TAU / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
        array[i] = vec2_angle(step * (float)i);

    simd_vector unit_epsilon = simd_splat(0.0005f);

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector vec_x, vec_y;
        simd_load_xy((float*)array + i * simd_vector_width, &vec_x, &vec_y);

        simd_vector approx = simd_vec2_approx_length(vec_x, vec_y);
        ASSERT(simd_all(simd_equal(approx, simd_splat(1.f), unit_epsilon)));

        simd_vector length = simd_splat((float) i + 1);
        vec_x = simd_mul(vec_x, length);
        vec_y = simd_mul(vec_y, length);

        approx = simd_vec2_approx_length(vec_x, vec_y);
        ASSERT(simd_all(simd_equal(approx, length, simd_mul(unit_epsilon, length))));
    }

    PASS();
}
