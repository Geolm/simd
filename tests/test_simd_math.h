
#include "../extra/vec2.h"
#include <float.h>


#define NUM_ELEMENTS (1024)
#define NUM_VECTORS (NUM_ELEMENTS/simd_vector_width)


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

    printf("simd_sincos max error : %.*e\n", FLT_DECIMAL_DIG, simd_hmax(max_error));

    PASS();
}

static inline float linear_to_srgb(float val)
{
	if (val <= 0.0031308f)
		return 12.92f * val;
	else
		return 1.055f * powf(val, 1.0f / 2.4f) - 0.055f;
}

static inline float srgb_to_linear(float val)
{
    if (val < 0.04045f)
        return val * (1.0f / 12.92f);
    else
        return powf((val + 0.055f) * (1.0f / 1.055f), 2.4f);
}

typedef float (*reference_function)(float);
typedef simd_vector (*approximation_function)(simd_vector);

TEST generic_test(reference_function ref, approximation_function approx, float range_min, float range_max, float epsilon, bool relative_error, const char* name)
{
    float input[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = ((range_max - range_min) / (float) (NUM_ELEMENTS-1));

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        input[i] = (step * (float)(i)) + range_min;
        result[i] = ref(input[i]);
    }

    simd_vector v_epsilon = simd_splat(epsilon);
    simd_vector v_max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector v_input = simd_load_offset(input, i);
        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = approx(v_input);

        simd_vector v_error = relative_error ? simd_div(simd_abs_diff(v_approx, v_result), v_result) : simd_abs_diff(v_approx, v_result);
        ASSERT(simd_all(simd_cmp_le(v_error, v_epsilon)));
        v_max_error = simd_max(v_max_error, v_error);
    }

    printf("%s max error : %.*e\n", name, FLT_DECIMAL_DIG, simd_hmax(v_max_error));
    
    PASS();
}

typedef float (*reference_function2)(float, float);
typedef simd_vector (*approximation_function2)(simd_vector, simd_vector);

TEST generic_test2(reference_function2 ref, approximation_function2 approx, float epsilon, bool relative_error, const char* name)
{
    vec2 array[NUM_ELEMENTS];
    float result[NUM_ELEMENTS];
    float step = VEC2_TAU / (float) (NUM_ELEMENTS-1);

    for(int i=0; i<NUM_ELEMENTS; ++i)
    {
        array[i] = vec2_scale(vec2_angle(step * (float)i), (float)(i+1));
        if (array[i].x == 0.f)
            array[i].x = 1.f;
        if (array[i].y == 0.f)
            array[i].y = 1.f;
        result[i] = ref(array[i].x, array[i].y);
    }

    simd_vector v_epsilon = simd_splat(epsilon);
    simd_vector v_max_error = simd_splat_zero();

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vector vec_x, vec_y;
        simd_load_xy((float*)array + i * simd_vector_width * 2, &vec_x, &vec_y);

        simd_vector v_result = simd_load_offset(result, i);
        simd_vector v_approx = approx(vec_x, vec_y);

        simd_vector v_error = relative_error ? simd_div(simd_abs_diff(v_approx, v_result), v_result) : simd_abs_diff(v_approx, v_result);
        ASSERT(simd_all(simd_cmp_le(v_error, v_epsilon)));
        v_max_error = simd_max(v_max_error, v_error);
    }

    printf("%s max error : %.*e\n", name, FLT_DECIMAL_DIG, simd_hmax(v_max_error));

    PASS();
}

float atan2_xy(float x, float y) {return atan2f(y, x);}

SUITE(trigonometry)
{
    printf(".");
    RUN_TESTp(generic_test, sinf, simd_sin, -10.f, 10.f, FLT_EPSILON, false, "simd_sin");
    RUN_TESTp(generic_test, cosf, simd_cos, -10.f, 10.f, FLT_EPSILON, false, "simd_cos");
    RUN_TESTp(generic_test, sinf, simd_approx_sin, -SIMD_MATH_TAU, SIMD_MATH_TAU, 2.e-06f, false, "simd_approx_sin");
    RUN_TESTp(generic_test, cosf, simd_approx_cos, -SIMD_MATH_TAU, SIMD_MATH_TAU, 2.e-06f, false, "simd_approx_cos");
    RUN_TESTp(generic_test, acosf, simd_acos, -1.f, 1.f, 1.e-06f, false, "simd_acos");
    RUN_TESTp(generic_test, asinf, simd_asin, -1.f, 1.f, 1.e-06f, false, "simd_asin");
    RUN_TESTp(generic_test, atanf, simd_atan, -10.f, 10.f, 1.e-04f, false, "simd_atan");
    RUN_TESTp(generic_test, acosf, simd_approx_acos, -1.f, 1.f, 1.e-04f, false, "simd_approx_arcos");
    RUN_TESTp(generic_test2, atan2_xy, simd_atan2, 1.e-06f, false, "simd_atan2");
    RUN_TEST(sinuscosinus);
}

SUITE(exponentiation)
{
    printf(".");
    RUN_TESTp(generic_test, logf, simd_log, FLT_EPSILON, 1000.f, 1.e-06f, false, "simd_log");
    RUN_TESTp(generic_test, expf, simd_exp, -87.f, 87.f, 1.e-06f, true, "simd_exp");
    RUN_TESTp(generic_test, expf, simd_approx_exp, -87.f, 87.f, 2.e-03f, true, "simd_approx_exp");
    RUN_TESTp(generic_test, cbrtf, simd_cbrt, -100.f, 100.f, 5.e-07f, false, "simd_cbrt");
}

SUITE(color_space)
{
    printf(".");
    RUN_TESTp(generic_test, srgb_to_linear, simd_approx_srgb_to_linear, 0.f, 1.f, 1.e-04f, false, "simd_approx_srgb_to_linear");
    RUN_TESTp(generic_test, linear_to_srgb, simd_approx_linear_to_srgb, 0.f, 1.f, 4.e-03f, false, "simd_approx_linear_to_srgb");
}

