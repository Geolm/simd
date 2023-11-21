#ifndef __SIMD_2D_COLLISION__
#define __SIMD_2D_COLLISION__

#include "vec2.h"
#include <stdint.h>

struct simdcol_context;

typedef void (*simdcol_intersection_callback)(void*, uint32_t);

enum flush_hint
{
    flush_aabb_triangle,
    flush_aabb_obb,
    flush_aabb_disc,
    flush_aabb_circle,
    flush_aabb_arc,
    flush_triangle_triangle,
    flush_triangle_disc,
    flush_segment_aabb,
    flush_segment_disc,
    flush_all
};


// mainly allocate internal buffers
struct simdcol_context* simdcol_init(void* user_context, simdcol_intersection_callback callback);
void simdcol_set_cb(struct simdcol_context* context, void* user_context, simdcol_intersection_callback callback);

// deferred intersection tests, when the batch is full we compute intersection tests and the callbacks are called if needed
void simdcol_aabb_triangle(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 p0, vec2 p1, vec2 p2);
void simdcol_aabb_obb(struct simdcol_context* context, uint32_t user_data, aabb box, segment obb_height, float obb_width);
void simdcol_aabb_disc(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 center, float radius);
void simdcol_aabb_circle(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 center, float radius, float width);
void simdcol_aabb_arc(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 center, float radius, float width, float orientation, float aperture);
void simdcol_triangle_triangle(struct simdcol_context* context, uint32_t user_data, const vec2 a[3], const vec2 b[3]);
void simdcol_segment_aabb(struct simdcol_context* context, uint32_t user_data, vec2 p0, vec2 p1, aabb box);
void simdcol_segment_disc(struct simdcol_context* context, uint32_t user_data, vec2 p0, vec2 p1, vec2 center, float radius);
void simdcol_triangle_disc(struct simdcol_context* context, uint32_t user_data, vec2 v0, vec2 v1, vec2 v2, circle disc);

// force to compute intersection right now, run callback on intersection
void simdcol_flush(struct simdcol_context* context, enum flush_hint hint);

// free buffers
void simdcol_terminate(struct simdcol_context* context);

#endif
