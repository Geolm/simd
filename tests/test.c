#include <stdio.h>
#include "../simd.h"

#define SOKOL_IMPL
#include "sokol_time.h"

#include "test_aabb.h"

int test_load_xy(void)
{
    float array[simd_vector_width*2];
    for(int i=0; i<simd_vector_width*2; ++i)
        array[i] = (float) i;
    
    printf("simd_load_xy :");
    
    simd_vector x, y;
    simd_load_xy(array, &x, &y);
    
    simd_store(array, x);
    simd_store(array+simd_vector_width, y);
    
    for(int i=0; i<2; ++i)
        for(int j=0; j<simd_vector_width; ++j)
            if (array[i*simd_vector_width + j] != (float)(i+(j*2)))
                return 0;
    
    printf(" ok\n");
    return 1;
}

int test_load_xyz(void)
{
    float array[simd_vector_width*3];
    for(int i=0; i<simd_vector_width*3; ++i)
        array[i] = (float) i;
    
    printf("simd_load_xyz :");
    
    simd_vector x, y, z;
    simd_load_xyz(array, &x, &y, &z);
    simd_store(array, x);
    simd_store(array+simd_vector_width, y);
    simd_store(array+simd_vector_width*2, z);
    
    for(int i=0; i<3; ++i)
        for(int j=0; j<simd_vector_width; ++j)
            if (array[i*simd_vector_width + j] != (float)(i+(j*3)))
                return 0;
    
    printf(" ok\n");
    return 1;
}

int test_sort(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (simd_vector_width-i);
    
    printf("simd_sort :");

    simd_vector a = simd_load(array);
    a = simd_sort(a);

    simd_store(array, a);
    
    for(int i=0; i<simd_vector_width-1; ++i)
        if (array[i] > array[i+1])
            return 0;

    simd_store(array, simd_reverse(a));

    for(int i=0; i<simd_vector_width-1; ++i)
        if (array[i] < array[i+1])
            return 0;

    printf(" ok\n");
    return 1;
}

int test_get_lane(void)
{
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
        array[i] = (float) (simd_vector_width-i);

    printf("simd_get_lane :");

    simd_vector a = simd_load(array);

    for(int i=0; i<simd_vector_width; ++i)
        if (simd_get_lane(a, i) != array[i])
            return 0;

    if (simd_get_first_lane(a) != array[0])
        return 0;

    printf(" ok\n");
    return 1;
}

int test_horizontal(void)
{
    float sum=0.0f;
    float array[simd_vector_width];
    for(int i=0; i<simd_vector_width; ++i)
    {
        array[i] = (float) i;
        sum += array[i];
    }

    printf("simd_horizontal :");

    simd_vector a = simd_load(array);
    if (simd_hsum(a) != sum)
        return 0;

    if (simd_hmin(a) != array[0])
        return 0;

    if (simd_hmax(a) != array[simd_last_lane])
        return 0;

    printf(" ok\n");
    return 1;
}


int main(int argc, const char * argv[])
{
    stm_setup();

    if (!test_load_xy())
        return -1;
    
    if (!test_load_xyz())
        return -1;
    
    if (!test_sort())
        return -1;

    if (!test_get_lane())
        return -1;

    if (!test_horizontal())
        return -1;
    
    //if (!test_aabb())
    //    return -1;
    
    return 0;
}
