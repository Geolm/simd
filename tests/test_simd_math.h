
#include "../vec2.h"
#include <float.h>

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

    simd_vector epsilon = simd_splat(0.057f);
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
        simd_vector v_approx = simd_atan(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));
        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf("simd_tan max error : %f\n", simd_hmax(max_error));

    PASS();
}

TEST arctan2(void)
{
    vec2 array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = VEC2_TAU / (float) (NUM_ELEMENTS+1);

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        result[i] = (step * (float)(i+1)) - VEC2_PI;
        array[i] = vec2_scale(vec2_angle(result[i]), (float)(i+1));
    }

    simd_vector epsilon = simd_splat(0.000003f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector vec_x, vec_y;
        simd_load_xy((float*)array + i * simd_vector_width * 2, &vec_x, &vec_y);

        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = simd_atan2(vec_x, vec_y);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));
        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf("simd_atan2 max error : %f\n", simd_hmax(max_error));

    PASS();
}

TEST approx_exp(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 174.f / (float) (NUM_ELEMENTS);

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - 87.f;
        result[i] = expf(array[i]);
    }

    simd_vector epsilon = simd_splat(0.002f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_array = simd_load_offset(array, i);
        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = simd_approx_exp(v_array);

        simd_vector relative_error = simd_div(simd_abs_diff(v_approx, v_result), v_result);
        ASSERT(simd_all(simd_cmp_lt(relative_error, epsilon)));
        max_error = simd_max(max_error, relative_error);
    }

    printf("simd_approx_exp max error : %f\n", simd_hmax(max_error));

    PASS();
}

TEST logarithm(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 1000.f / (float) (NUM_ELEMENTS);

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)(i+1));
        result[i] = logf(array[i]);
    }

    simd_vector epsilon = simd_splat(FLT_EPSILON);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_array = simd_load_offset(array, i);
        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = simd_log(v_array);

        simd_vector relative_error = simd_div(simd_abs_diff(v_approx, v_result), v_result);
        ASSERT(simd_all(simd_cmp_lt(relative_error, epsilon)));
        max_error = simd_max(max_error, relative_error);
    }

    printf("simd_log max error : %f\n", simd_hmax(max_error));

    PASS();
}

SUITE(trigonometry)
{
    RUN_TEST(sinus);
    RUN_TEST(approx_sin);
    RUN_TEST(arcos);
    RUN_TEST(approx_arcos);
    RUN_TEST(arctan);
    RUN_TEST(arctan2);
    RUN_TEST(approx_exp);
    RUN_TEST(logarithm);
}

