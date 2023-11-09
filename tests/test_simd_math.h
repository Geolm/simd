
#include "../vec2.h"
#include "../simd_vec2.h"

TEST sinus(void)
{
    float array[simd_vector_width];
    float result[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
    {
        array[i] = (float) (i - simd_vector_width/2);
        result[i] = sinf(array[i]);
    }

    simd_vector a = simd_load(array);
    simd_vector target = simd_load(result);
    
    simd_vector epsilon = simd_splat(0.000001f);
    
    ASSERT(simd_all(simd_equal(simd_sin(a), target, epsilon)));
    PASS();
}

TEST approx_sin(void)
{
    float array[simd_vector_width];
    float result[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
    {
        array[i] = (float) (i - simd_vector_width/2);
        result[i] = sinf(array[i]);
    }

    simd_vector a = simd_load(array);
    simd_vector target = simd_load(result);
    simd_vector epsilon = simd_splat(0.0003f);
    
    ASSERT(simd_all(simd_equal(simd_approx_sin(a), target, epsilon)));
    PASS();
}

TEST arcos(void)
{
    float array[simd_vector_width];
    float result[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
    {
        array[i] = (float) (i - simd_vector_width/2) / (float)(simd_vector_width/2);
        result[i] = acosf(array[i]);
    }
    
    simd_vector a = simd_load(array);
    simd_vector target = simd_load(result);
    simd_vector epsilon = simd_splat(0.02f);

    ASSERT(simd_all(simd_equal(simd_approx_acos(a), target, epsilon)));
    PASS();
}

SUITE(trigonometry)
{
    RUN_TEST(sinus);
    RUN_TEST(approx_sin);
    RUN_TEST(arcos);
}

#define NUM_VECTORS (100)
#define NUM_ELEMENTS (simd_vector_width * NUM_VECTORS)

TEST approx_length()
{
    vec2 array[NUM_ELEMENTS];
    float step = VEC2_TAU / (float) NUM_ELEMENTS;

    for(int i=0; i<NUM_ELEMENTS; ++i)
        array[i] = vec2_angle(step * (float)i);

    simd_vector unit_epsilon = simd_splat(0.0005f);
    simd_vector any_epsilon = simd_splat(0.03f);

    for(int i=0; i<NUM_VECTORS; ++i)
    {
        simd_vec2 vec;
        simd_load_xy((float*)array + i * simd_vector_width, &vec.x, &vec.y);

        simd_vector approx = simd_vec2_approx_length(vec);
        ASSERT(simd_all(simd_equal(approx, simd_splat(1.f), unit_epsilon)));

        simd_vector length = simd_splat((float) i + 1);
        vec.x = simd_mul(vec.x, length);
        vec.y = simd_mul(vec.y, length);

        approx = simd_vec2_approx_length(vec);
        ASSERT(simd_all(simd_equal(approx, length, any_epsilon)));
    }

    PASS();
}
