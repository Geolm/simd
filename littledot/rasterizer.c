#include "rasterizer.h"
#include <stdlib.h>
#include "../simd.h"

struct lines_array
{
    float* x0;
    float* y0;
    float* x1;
    float* y1;
    float* width;
    float* red;
    float* green;
    float* blue;

    int num_lines;
    int max_lines;
};

#define TILE_WIDTH (16)
#define NUM_PIXELS_PER_TILE (TILE_WIDTH*TILE_WIDTH)

struct tile
{
    struct lines_array lines;

    float red[NUM_PIXELS_PER_TILE];
    float green[NUM_PIXELS_PER_TILE];
    float blue[NUM_PIXELS_PER_TILE];
    float alpha[NUM_PIXELS_PER_TILE];

    int top, left;
};


struct tiles_array
{
    struct tile* tiles;

    float* aabb_x_min;
    float* aabb_y_min;
    float* aabb_x_max;
    float* aabb_y_max;

    int num_tiles;
    int max_tiles;
};

struct rasterizer_context
{
    struct tiles_array tiles;

    int width;
    int height;
    int num_pixels;
    float max_u;
    float max_v;
    float uv_to_xy;
    float pixel_size;
};


//----------------------------------------------------------------------------------------------------------------------
struct rasterizer_context* rasterizer_init(int width, int height)
{
    struct rasterizer_context* context = (struct rasterizer_context*) malloc(sizeof(struct rasterizer_context));

    context->width = width;
    context->height = height;
    context->num_pixels = width * height;

    if (width>height)
    {
        context->max_u = 1.f;
        context->max_v = (float) height / (float) width;
        context->uv_to_xy = (float) (width - 1);
    }
    else
    {
        context->max_v = 1.f;
        context->max_u = (float) width / (float) height;
        context->uv_to_xy = (float) (height - 1);
    }
    context->pixel_size = 1.f/context->uv_to_xy;

    return context;
}

//----------------------------------------------------------------------------------------------------------------------
static inline simd_vector slab_test(simd_vector slab_min, simd_vector slab_max,
                                   simd_vector segment_start, simd_vector segment_end, simd_vector inv_dir)
{
    simd_vector t0 = simd_mul(simd_sub(slab_min, segment_start), inv_dir);
    simd_vector t1 = simd_mul(simd_sub(slab_max, segment_end), inv_dir);
    simd_vector enter = simd_min(t0, t1);
    simd_vector exit = simd_max(t0, t1);
    
    simd_vector lt = simd_cmp_lt(enter, simd_splat(1.f));
    simd_vector gt = simd_cmp_gt(exit, simd_splat_zero());

    return simd_and(gt, lt);
}

//----------------------------------------------------------------------------------------------------------------------
void rasterizer_cull_lines(struct tiles_array* tiles, struct lines_array* lines)
{
    int num_vec = lines->num_lines / simd_vector_width;
    int remaining_lines = lines->num_lines - (num_vec * simd_vector_width);

    for(int tile_index=0; tile_index<tiles->num_tiles; ++tile_index)
    {
        for(int vec_index=0; vec_index<num_vec; ++vec_index)
        {
            int offset = vec_index * simd_vector_width;
            simd_vector x0 = simd_load(lines->x0 + offset);
            simd_vector x1 = simd_load(lines->x1 + offset);
            simd_vector y0 = simd_load(lines->y0 + offset);
            simd_vector y1 = simd_load(lines->y1 + offset);
            simd_vector half_width = simd_mul(simd_load(lines->width + offset), simd_splat(0.5f));
            simd_vector inv_dir_x = simd_rcp(simd_sub(x1, x0));
            simd_vector inv_dir_y = simd_rcp(simd_sub(y1, y0));
            simd_vector aabb_x_min = simd_sub(simd_splat(tiles->aabb_x_min[tile_index]), half_width);
            simd_vector aabb_x_max = simd_add(simd_splat(tiles->aabb_x_max[tile_index]), half_width);
            simd_vector aabb_y_min = simd_sub(simd_splat(tiles->aabb_y_min[tile_index]), half_width);
            simd_vector aabb_y_max = simd_add(simd_splat(tiles->aabb_y_max[tile_index]), half_width);

            simd_vector intersection = simd_and(slab_test(aabb_x_min, aabb_x_max, x0, x1, inv_dir_x),
                                                slab_test(aabb_y_min, aabb_y_max, y0, y1, inv_dir_y));
            
            int bits = simd_get_mask(intersection);
            for(int i=0; i<simd_vector_width; ++i)
            {
                if (bits&i)
                {
                    int index = offset+i;
                    lines_array_add(&tiles->tiles[tile_index], lines->x0[index], lines->y0[index],
                                    lines->x1[index], lines->y1[index], lines->width[index]);
                }
            }
        }   
    }
}

//----------------------------------------------------------------------------------------------------------------------
void lines_array_add(struct lines_array* lines, float x0, float y0, float x1, float y1, float width)
{
    
}
