#include "../simd.h"
#include <float.h>

typedef struct {float x, y, z;} point;

void compute_aabb(const point* points, int num_points, point* aabb_min, point* aabb_max)
{
    int num_vec = num_points / simd_vector_width;
    int remaining_points = num_points - (num_vec * simd_vector_width);
    
    simd_vector min_x = simd_splat(FLT_MAX); simd_vector min_y = min_x; simd_vector min_z = min_x;
    simd_vector max_x = simd_splat(-FLT_MAX); simd_vector max_y = max_x; simd_vector max_z = max_x;
    
    for(int i=0; i<num_vec; ++i)
    {
        simd_vector x, y, z;
        simd_load_xyz_unorder((const float*)points, &x, &y, &z);
        
        min_x = simd_min(min_x, x);
        min_y = simd_min(min_y, y);
        min_z = simd_min(min_z, z);
        max_x = simd_max(max_x, x);
        max_y = simd_max(max_y, y);
        max_z = simd_max(max_z, z);
        
        points += simd_vector_width;
    }
    
    min_x = simd_sort(min_x); aabb_min->x = simd_get_lane(min_x, 0);
    min_y = simd_sort(min_y); aabb_min->y = simd_get_lane(min_y, 0);
    min_z = simd_sort(min_z); aabb_min->z = simd_get_lane(min_z, 0);
    
    max_x = simd_sort(max_x); aabb_max->x = simd_get_lane(max_x, simd_vector_width-1);
    max_y = simd_sort(max_y); aabb_max->y = simd_get_lane(max_y, simd_vector_width-1);
    max_z = simd_sort(max_z); aabb_max->z = simd_get_lane(max_z, simd_vector_width-1);
    
    for(int i=0; i<remaining_points; ++i)
    {
        if (points[i].x<aabb_min->x)
            aabb_min->x = points[i].x;
        if (points[i].y<aabb_min->y)
            aabb_min->y = points[i].y;
        if (points[i].z<aabb_min->z)
            aabb_min->z = points[i].z;
        
        if (points[i].x>aabb_max->x)
            aabb_max->x = points[i].x;
        if (points[i].y>aabb_max->y)
            aabb_max->y = points[i].y;
        if (points[i].z>aabb_max->z)
            aabb_max->z = points[i].z;
    }
}
