
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

    simd_vector epsilon = simd_splat(FLT_EPSILON);
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

TEST sinuscosinus(void)
{
    float array[NUM_ELEMENTS];
    float result_cos[NUM_ELEMENTS];
    float result_sin[NUM_ELEMENTS];
    float step = 8192.f / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - VEC2_PI;
        result_cos[i] = cosf(array[i]);
        result_sin[i] = sinf(array[i]);
    }

    simd_vector epsilon = simd_splat(FLT_EPSILON);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_result_cos = simd_load_offset(result_cos, i);
        simd_vector v_result_sin = simd_load_offset(result_sin, i);

        simd_vector approx_sin, approx_cos;
        simd_sincos(simd_load_offset(array,  i), &approx_sin, &approx_cos);

        ASSERT(simd_all(simd_equal(approx_sin, v_result_sin, epsilon)));
        ASSERT(simd_all(simd_equal(approx_cos, v_result_cos, epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(approx_sin, v_result_sin));
        max_error = simd_max(max_error, simd_abs_diff(approx_cos, v_result_cos));
    }

    printf("simd_sincos max error : %f\n", simd_hmax(max_error));

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

    simd_vector epsilon = simd_splat(0.000002f);
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

TEST arcsin(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 2.f / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - 1.f;
        result[i] = asinf(array[i]);
    }

    simd_vector epsilon = simd_splat(FLT_MAX);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_approx = simd_asin(simd_load_offset(array,  i));
        simd_vector v_result = simd_load_offset(result, i);

        ASSERT(simd_all(simd_equal(v_approx, v_result, epsilon)));

        max_error = simd_max(max_error, simd_abs_diff(v_approx, v_result));
    }

    printf("simd_asin max error : %f\n", simd_hmax(max_error));

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

    printf(".simd_log max error : %f\n", simd_hmax(max_error));

    PASS();
}

TEST exponential(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 174.f / (float) (NUM_ELEMENTS);

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)i) - 87.f;
        result[i] = expf(array[i]);
    }

    simd_vector epsilon = simd_splat(FLT_EPSILON);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_array = simd_load_offset(array, i);
        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = simd_exp(v_array);

        simd_vector relative_error = simd_div(simd_abs_diff(v_approx, v_result), v_result);
        ASSERT(simd_all(simd_cmp_lt(relative_error, epsilon)));
        max_error = simd_max(max_error, relative_error);
    }

    printf("simd_exp max error : %f\n", simd_hmax(max_error));

    PASS();
}

static inline float linear_to_srgb(float val)
{
	if (val <= 0.0031308f)
		return 12.92f * val;
	else
		return 1.055f * powf(val, 1.0f / 2.4f) - 0.055f;
}

TEST linear_to_srgb_test(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 1.f / (float) (NUM_ELEMENTS);

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)(i));
        result[i] = linear_to_srgb(array[i]);
    }

    simd_vector epsilon = simd_splat(0.004f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_array = simd_load_offset(array, i);
        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = simd_approx_linear_to_srgb(v_array);

        simd_vector error = simd_abs_diff(v_approx, v_result);
        ASSERT(simd_all(simd_cmp_lt(error, epsilon)));
        max_error = simd_max(max_error, error);
    }

    printf(".simd_approx_linear_to_srgb max error : %f\n", simd_hmax(max_error));
    
    PASS();
}

static inline float srgb_to_linear(float val)
{
    if (val < 0.04045f)
        return val * (1.0f / 12.92f);
    else
        return powf((val + 0.055f) * (1.0f / 1.055f), 2.4f);
}

TEST srgb_to_linear_test(void)
{
    float array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = 1.f / (float) (NUM_ELEMENTS);

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = (step * (float)(i));
        result[i] = srgb_to_linear(array[i]);
    }

    simd_vector epsilon = simd_splat(0.00008f);
    simd_vector max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_array = simd_load_offset(array, i);
        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = simd_approx_srgb_to_linear(v_array);

        simd_vector error = simd_abs_diff(v_approx, v_result);
        ASSERT(simd_all(simd_cmp_lt(error, epsilon)));
        max_error = simd_max(max_error, error);
    }

    printf("simd_approx_srgb_to_linear max error : %f\n", simd_hmax(max_error));
    
    PASS();
}


SUITE(trigonometry)
{
    RUN_TEST(sinus);
    RUN_TEST(approx_sin);
    RUN_TEST(arcos);
    RUN_TEST(arcsin);
    RUN_TEST(approx_arcos);
    RUN_TEST(arctan2);
    RUN_TEST(sinuscosinus);
}

SUITE(exponentiation)
{
    RUN_TEST(logarithm);
    RUN_TEST(exponential);
    RUN_TEST(approx_exp);
}

SUITE(color_space)
{
    RUN_TEST(linear_to_srgb_test);
    RUN_TEST(srgb_to_linear_test);
}

