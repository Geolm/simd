


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
