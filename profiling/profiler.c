#define SOKOL_TIME_IMPL
#include "sokol_time.h"
#include "../simd_math.h"
#include "float.h"
#include "stdio.h"

#define A_LOT (4000000000)

//-----------------------------------------------------------------------------
float compare_sinus(void)
{
    float init_array[simd_vector_width];

    for(uint32_t i=0; i<simd_vector_width; ++i)
        init_array[i] = (float) (i) / (float) (simd_vector_width);

    simd_vector step = simd_splat(FLT_EPSILON);
    simd_vector angle = simd_load(init_array);
    simd_vector result = simd_splat_zero();

    printf("- comparing sinus functions :\n"); uint64_t start = stm_now();

    for(uint32_t i=0; i<A_LOT; ++i)
    {
        result = simd_add(result, simd_sin(angle));
        angle = simd_add(angle, step);
    }

    printf("  simd_sinus %3.3f ms \n", stm_ms(stm_since(start)));

    start = stm_now();

    for(uint32_t i=0; i<A_LOT; ++i)
    {
        result = simd_add(result, simd_approx_sin(angle));
        angle = simd_add(angle, step);
    }

    printf("  simd_approx_sin %3.3f ms \n", stm_ms(stm_since(start)));

    return simd_hmax(result);
}

static simd_vector simd_vec2_length(simd_vector x, simd_vector y) {return simd_sqrt(simd_fmad(x, x, simd_mul(y, y))); }

//-----------------------------------------------------------------------------
float compare_vec_length(void)
{
    float init_array[simd_vector_width];

    for(uint32_t i=0; i<simd_vector_width; ++i)
        init_array[i] = (float) (i) / (float) (simd_vector_width);

    simd_vector step = simd_splat(1.f);
    simd_vector x = simd_load(init_array);
    simd_vector y = simd_neg(x);
    simd_vector result = simd_splat_zero();

    printf("- comparing vec2 length functions :\n"); uint64_t start = stm_now();

    for(uint32_t i=0; i<A_LOT; ++i)
    {
        result = simd_add(result, simd_vec2_length(x, y));
        x = simd_add(x, simd_splat(1.f));
        y = simd_add(y, simd_splat(-2.f));
    }

    printf("  simd_vec2_length %3.3f ms \n", stm_ms(stm_since(start))); start = stm_now();

    for(uint32_t i=0; i<A_LOT; ++i)
    {
        result = simd_add(result, simd_vec2_approx_length(x, y));
        x = simd_add(x, simd_splat(1.f));
        y = simd_add(y, simd_splat(-2.f));
    }

    printf("  simd_vec2_approx_length %3.3f ms \n", stm_ms(stm_since(start))); start = stm_now();

    return simd_hmax(result);

}

int main(int argc, char * argv[])
{
    stm_setup();

    int output = 0;
    
    output += (int) compare_sinus();
    output += (int) compare_vec_length();

    return output;
}