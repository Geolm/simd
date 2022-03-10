#include "../simd.h"
#include <float.h>
#include "fast_obj.h"
#include "sokol_time.h"
#include <stdio.h>

typedef struct {float x, y, z;} point;

#define _min(a, b) ((a<b) ? a : b)
#define _max(a, b) ((a>b) ? a : b)

void simd_compute_aabb(const point* points, int num_points, point* aabb_min, point* aabb_max)
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
    
    aabb_min->x = simd_hmin(min_x);
    aabb_min->y = simd_hmin(min_y);
    aabb_min->z = simd_hmin(min_z);
    
    aabb_max->x = simd_hmax(max_x);
    aabb_max->y = simd_hmax(max_y);
    aabb_max->z = simd_hmax(max_z);
    
    for(int i=0; i<remaining_points; ++i)
    {
        aabb_min->x = _min(aabb_min->x, points[i].x);
        aabb_min->y = _min(aabb_min->y, points[i].y);
        aabb_min->z = _min(aabb_min->z, points[i].z);
        aabb_max->x = _max(aabb_max->x, points[i].x);
        aabb_max->y = _max(aabb_max->y, points[i].y);
        aabb_max->z = _max(aabb_max->z, points[i].z);
    }
}

void compute_aabb(const point* points, int num_points, point* aabb_min, point* aabb_max)
{
    aabb_min->x = aabb_min->y = aabb_min->z = FLT_MAX;
    aabb_max->x = aabb_max->y = aabb_max->z = -FLT_MAX;
    
    for(int i=0; i<num_points; ++i)
    {
        aabb_min->x = _min(aabb_min->x, points[i].x);
        aabb_min->y = _min(aabb_min->y, points[i].y);
        aabb_min->z = _min(aabb_min->z, points[i].z);
        aabb_max->x = _max(aabb_max->x, points[i].x);
        aabb_max->y = _max(aabb_max->y, points[i].y);
        aabb_max->z = _max(aabb_max->z, points[i].z);
    }
}

int test_aabb(void)
{
    fastObjMesh* mesh = fast_obj_read("/Users/geolm/Documents/meshes/mustang.obj");

    point non_simd_aabb_min, non_simd_aabb_max;
    
    uint64_t last_time = stm_now();

    compute_aabb((point*)mesh->positions, mesh->position_count, &non_simd_aabb_min, &non_simd_aabb_max);

    uint64_t delta_time = stm_laptime(&last_time);

    printf("test_aabb\n  non-simd aabb computation in %fms\n", stm_ms(delta_time));

    point aabb_min, aabb_max;

    simd_compute_aabb((point*)mesh->positions, mesh->position_count, &aabb_min, &aabb_max);

    delta_time = stm_laptime(&last_time);

    printf("  simd aabb computation in %fms\n", stm_ms(delta_time));

    //mesh->positions

    fast_obj_destroy(mesh);
    
    if (aabb_min.x != non_simd_aabb_min.x ||
        aabb_min.y != non_simd_aabb_min.y ||
        aabb_min.z != non_simd_aabb_min.z ||
        aabb_max.x != non_simd_aabb_max.x ||
        aabb_max.y != non_simd_aabb_max.y ||
        aabb_max.z != non_simd_aabb_max.z)
        return 0;

    return 1;
}
