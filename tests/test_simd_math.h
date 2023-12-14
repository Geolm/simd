
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
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_sin(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf(".simd_sin max error : %f\n", simd_hmax(max_error));

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

    simd_vector epsilon = simd_splat(0.0011f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_approx_sin(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf("simd_approx_sin max error : %f\n", simd_hmax(max_error));

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

    simd_vector epsilon = simd_splat(0.00007f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_acos(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf("simd_acos max error : %f\n", simd_hmax(max_error));

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
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_approx_acos(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf("simd_approx_acos max error : %f\n", simd_hmax(max_error));

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

    simd_vector epsilon = simd_splat(0.000003f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_approx_atan(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf("simd_approx_tan max error : %f\n", simd_hmax(max_error));

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

    simd_vector unit_epsilon = simd_splat(0.00006f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector vec_x, vec_y;
        simd_load_xy((float*)array + i * simd_vector_width, &vec_x, &vec_y);

        simd_vector approx = simd_vec2_approx_length(vec_x, vec_y);
        ASSERT(simd_all(simd_equal(approx, simd_splat(1.f), unit_epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(approx, simd_splat(1.f)));

        simd_vector length = simd_splat((float) i + 1);
        vec_x = simd_mul(vec_x, length);
        vec_y = simd_mul(vec_y, length);

        approx = simd_vec2_approx_length(vec_x, vec_y);
        ASSERT(simd_all(simd_equal(approx, length, simd_mul(unit_epsilon, length))));
    }

    printf("simd_vec2_approx_length max error : %f\n", simd_hmax(max_error));

    PASS();
}
