#include "simd_2d_collision.h"
#include "simd.h"
#include "simd_math.h"
#include <stddef.h>

// SoA, Structure of Array, Vec2
typedef struct
{
    simd_vector x;
    simd_vector y;
} simd_vec2;


static inline simd_vec2 simd_vec2_load(const float* x, const float* y, ptrdiff_t offset) {return (simd_vec2) {.x = simd_load(x + offset), .y = simd_load(y + offset)};}
static inline simd_vec2 simd_vec2_add(simd_vec2 a, simd_vec2 b) {return (simd_vec2) {.x = simd_add(a.x, b.x), .y = simd_add(a.y, b.y)};}
static inline simd_vec2 simd_vec2_sub(simd_vec2 a, simd_vec2 b) {return (simd_vec2) {.x = simd_sub(a.x, b.x), .y = simd_sub(a.y, b.y)};}
static inline simd_vec2 simd_vec2_mul(simd_vec2 a, simd_vec2 b) {return (simd_vec2) {.x = simd_mul(a.x, b.x), .y = simd_mul(a.y, b.y)};}
static inline simd_vec2 simd_vec2_scale(simd_vec2 a, simd_vector scale) {return (simd_vec2) {.x = simd_mul(a.x, scale), .y = simd_mul(a.y, scale)};}
static inline simd_vec2 simd_vec2_abs(simd_vec2 a) {return (simd_vec2) {.x = simd_abs(a.x), .y = simd_abs(a.y)};}
static inline simd_vec2 simd_vec2_max(simd_vec2 a, simd_vec2 b) {return (simd_vec2) {.x = simd_max(a.x, b.x), .y = simd_max(a.y, b.y)};}
static inline simd_vec2 simd_vec2_clamp(simd_vec2 a, simd_vec2 range_min, simd_vec2 range_max) 
{
    return (simd_vec2) {.x = simd_clamp(a.x, range_min.x, range_max.x), .y = simd_clamp(a.y, range_min.y, range_max.y)};
}
static inline simd_vector simd_vec2_dot(simd_vec2 a, simd_vec2 b) {return simd_fmad(a.x, b.x, simd_mul(a.y, b.y));}
static inline simd_vector simd_vec2_sq_length(simd_vec2 a) {return simd_vec2_dot(a, a);}
static inline simd_vec2 simd_vec2_normalize(simd_vec2 a)
{
    simd_vector rcp_length = simd_rsqrt(simd_vec2_sq_length(a));
    return (simd_vec2) {.x = simd_mul(a.x, rcp_length), .y = simd_mul(a.y, rcp_length)};
}


//-----------------------------------------------------------------------------
#define BATCH_SIZE (simd_vector_width*4)
#define BATCH_NUM_VECTOR (BATCH_SIZE/simd_vector_width)

//-----------------------------------------------------------------------------
struct aabb_triangle_data
{
    float aabb_min_x[BATCH_SIZE];
    float aabb_min_y[BATCH_SIZE];
    float aabb_max_x[BATCH_SIZE];
    float aabb_max_y[BATCH_SIZE];

    // triangle vertices
    float v_x[3][BATCH_SIZE];
    float v_y[3][BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct aabb_obb_data
{
    float aabb_min_x[BATCH_SIZE];
    float aabb_min_y[BATCH_SIZE];
    float aabb_max_x[BATCH_SIZE];
    float aabb_max_y[BATCH_SIZE];

    // obb normalized axis
    float j_a[BATCH_SIZE];
    float j_b[BATCH_SIZE];
    float j_c[BATCH_SIZE];
    float i_c[BATCH_SIZE];
    float extent_i[BATCH_SIZE];
    float extent_j[BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct aabb_disc_data
{
    float aabb_min_x[BATCH_SIZE];
    float aabb_min_y[BATCH_SIZE];
    float aabb_max_x[BATCH_SIZE];
    float aabb_max_y[BATCH_SIZE];

    float center_x[BATCH_SIZE];
    float center_y[BATCH_SIZE];
    float sq_radius[BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct aabb_circle_data
{
    float aabb_min_x[BATCH_SIZE];
    float aabb_min_y[BATCH_SIZE];
    float aabb_max_x[BATCH_SIZE];
    float aabb_max_y[BATCH_SIZE];

    float center_x[BATCH_SIZE];
    float center_y[BATCH_SIZE];
    float sq_outter_radius[BATCH_SIZE];
    float sq_inner_radius[BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct aabb_arc_data
{
    float aabb_min_x[BATCH_SIZE];
    float aabb_min_y[BATCH_SIZE];
    float aabb_max_x[BATCH_SIZE];
    float aabb_max_y[BATCH_SIZE];

    float center_x[BATCH_SIZE];
    float center_y[BATCH_SIZE];
    float outter_radius[BATCH_SIZE];
    float sq_inner_radius[BATCH_SIZE];
    float orientation[BATCH_SIZE];
    float aperture[BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct triangle_triangle_data
{
    float tri0_x[3][BATCH_SIZE];
    float tri0_y[3][BATCH_SIZE];
    float tri1_x[3][BATCH_SIZE];
    float tri1_y[3][BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct segment_aabb_data
{
    float p0_x[BATCH_SIZE];
    float p0_y[BATCH_SIZE];
    float p1_x[BATCH_SIZE];
    float p1_y[BATCH_SIZE];

    float aabb_min_x[BATCH_SIZE];
    float aabb_min_y[BATCH_SIZE];
    float aabb_max_x[BATCH_SIZE];
    float aabb_max_y[BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct segment_disc_data
{
    float p0_x[BATCH_SIZE];
    float p0_y[BATCH_SIZE];
    float p1_x[BATCH_SIZE];
    float p1_y[BATCH_SIZE];

    float center_x[BATCH_SIZE];
    float center_y[BATCH_SIZE];
    float sq_radius[BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct triangle_disc_data
{
    float v_x[3][BATCH_SIZE];
    float v_y[3][BATCH_SIZE];

    float center_x[BATCH_SIZE];
    float center_y[BATCH_SIZE];
    float sq_radius[BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
struct point_triangle_data
{
    float p_x[BATCH_SIZE];
    float p_y[BATCH_SIZE];

    float v_x[3][BATCH_SIZE];
    float v_y[3][BATCH_SIZE];

    uint32_t user_data[BATCH_SIZE];
    uint32_t num_items;
};

//-----------------------------------------------------------------------------
enum simdcol_state
{
    state_idle,
    state_processing
};

//-----------------------------------------------------------------------------
struct simdcol_context
{
    struct aabb_triangle_data* aabb_triangle;
    struct aabb_obb_data* aabb_obb;
    struct aabb_disc_data* aabb_disc;
    struct aabb_circle_data* aabb_circle;
    struct aabb_arc_data* aabb_arc;
    struct triangle_triangle_data* triangle_triangle;
    struct segment_aabb_data* segment_aabb;
    struct segment_disc_data* segment_disc;
    struct triangle_disc_data* triangle_disc;
    struct point_triangle_data* point_triangle;

    void* user_context;
    simdcol_intersection_callback on_intersection;
    enum simdcol_state state;
};

//-----------------------------------------------------------------------------
struct simdcol_context* simdcol_init(void* user_context, simdcol_intersection_callback callback)
{
    assert(callback != NULL);

    struct simdcol_context* context = (struct simdcol_context*) malloc(sizeof(struct simdcol_context));
    context->aabb_triangle = (struct aabb_triangle_data*) simd_aligned_alloc(sizeof(struct aabb_triangle_data));
    context->aabb_obb = (struct aabb_obb_data*) simd_aligned_alloc(sizeof(struct aabb_obb_data));
    context->aabb_disc = (struct aabb_disc_data*) simd_aligned_alloc(sizeof(struct aabb_disc_data));
    context->aabb_circle = (struct aabb_circle_data*) simd_aligned_alloc(sizeof(struct aabb_circle_data));
    context->aabb_arc = (struct aabb_arc_data*) simd_aligned_alloc(sizeof(struct aabb_arc_data));
    context->triangle_triangle = (struct triangle_triangle_data*) simd_aligned_alloc(sizeof(struct triangle_triangle_data));
    context->segment_aabb = (struct segment_aabb_data*) simd_aligned_alloc(sizeof(struct segment_aabb_data));
    context->segment_disc = (struct segment_disc_data*) simd_aligned_alloc(sizeof(struct segment_disc_data));
    context->triangle_disc = (struct triangle_disc_data*) simd_aligned_alloc(sizeof(struct triangle_disc_data));
    context->point_triangle = (struct point_triangle_data*) simd_aligned_alloc(sizeof(struct point_triangle_data));

    context->aabb_triangle->num_items = 0;
    context->aabb_obb->num_items = 0;
    context->aabb_disc->num_items = 0;
    context->aabb_circle->num_items = 0;
    context->aabb_arc->num_items = 0;
    context->triangle_triangle->num_items = 0;
    context->segment_aabb->num_items = 0;
    context->segment_disc->num_items = 0;
    context->triangle_disc->num_items = 0;
    context->point_triangle->num_items = 0;

    context->state = state_idle;
    context->on_intersection = callback;
    context->user_context = user_context;
    return context;
}

//-----------------------------------------------------------------------------
void simdcol_set_cb(struct simdcol_context* context, void* user_context, simdcol_intersection_callback callback)
{
    simdcol_flush(context, flush_all);
    context->on_intersection = callback;
    context->user_context = user_context;
}

//-----------------------------------------------------------------------------
void simdcol_aabb_triangle(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 p0, vec2 p1, vec2 p2)
{
    assert(context->aabb_triangle->num_items < BATCH_SIZE);
    assert(context->state == state_idle);    // can't add intersection tests from the callback

    struct aabb_triangle_data* data = context->aabb_triangle;
    unsigned int index = data->num_items++;

    // copy data in SoA
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->v_x[0][index] = p0.x;
    data->v_x[1][index] = p1.x;
    data->v_x[2][index] = p2.x;
    data->v_y[0][index] = p0.y;
    data->v_y[1][index] = p1.y;
    data->v_y[2][index] = p2.y;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_aabb_triangle);
}

//-----------------------------------------------------------------------------
void simdcol_aabb_obb(struct simdcol_context* context, uint32_t user_data, aabb box, segment obb_height, float obb_width)
{
    assert(context->aabb_triangle->num_items < BATCH_SIZE);
    assert(context->state == state_idle);

    struct aabb_obb_data* data = context->aabb_obb;
    uint32_t index = data->num_items++;

    vec2 center = vec2_scale(vec2_add(obb_height.p0, obb_height.p1), .5f);
    vec2 dir = vec2_sub(obb_height.p1, obb_height.p0);
    
    data->extent_j[index] = vec2_normalize(&dir) * .5f;
    data->j_a[index] = dir.x;
    data->j_b[index] = dir.y;
    data->j_c[index] = -vec2_dot(dir, center);
    data->extent_i[index] = obb_width * .5f;
    data->i_c[index] = -vec2_dot(vec2_skew(dir), center);
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_aabb_obb);
}

//-----------------------------------------------------------------------------
void simdcol_aabb_disc(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 center, float radius)
{
    assert(context->aabb_disc->num_items < BATCH_SIZE);
    assert(context->state == state_idle);

    struct aabb_disc_data* data = context->aabb_disc;
    uint32_t index = data->num_items++;

    // copy data in SoA
    data->user_data[index] = user_data;
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->center_x[index] = center.x;
    data->center_y[index] = center.y;
    data->sq_radius[index] = radius * radius;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_aabb_disc);
}

//-----------------------------------------------------------------------------
void simdcol_aabb_circle(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 center, float radius, float width)
{
    assert(context->aabb_circle->num_items < BATCH_SIZE);
    assert(context->state == state_idle);

    struct aabb_circle_data* data = context->aabb_circle;
    uint32_t index = data->num_items++;

    float half_width = width * .5f;

    // copy data in SoA
    data->user_data[index] = user_data;
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->center_x[index] = center.x;
    data->center_y[index] = center.y;
    data->sq_outter_radius[index] = float_square(radius + half_width);
    data->sq_inner_radius[index] = float_square(radius - half_width);

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_aabb_circle);
}

//-----------------------------------------------------------------------------
void simdcol_aabb_arc(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 center, float radius, float width, float orientation, float aperture)
{
    assert(context->aabb_arc->num_items < BATCH_SIZE);
    assert(context->state == state_idle);

    struct aabb_arc_data* data = context->aabb_arc;
    uint32_t index = data->num_items++;

    float half_width = width * .5f;

    data->user_data[index] = user_data;
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->center_x[index] = center.x;
    data->center_y[index] = center.y;
    data->outter_radius[index] = radius + half_width;
    data->sq_inner_radius[index] = float_square(radius - half_width);
    data->orientation[index] = orientation;
    data->aperture[index] = aperture;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_aabb_arc);
}

//-----------------------------------------------------------------------------
void simdcol_triangle_triangle(struct simdcol_context* context, uint32_t user_data, const vec2 a[3], const vec2 b[3])
{
    assert(context->triangle_triangle->num_items < BATCH_SIZE);
    assert(context->state == state_idle);

    struct triangle_triangle_data* data = context->triangle_triangle;
    uint32_t index = data->num_items++;

    data->user_data[index] = user_data;
    data->tri0_x[0][index] = a[0].x;
    data->tri0_x[1][index] = a[1].x;
    data->tri0_x[2][index] = a[2].x;
    data->tri0_y[0][index] = a[0].y;
    data->tri0_y[1][index] = a[1].y;
    data->tri0_y[2][index] = a[2].y;
    data->tri1_x[0][index] = b[0].x;
    data->tri1_x[1][index] = b[1].x;
    data->tri1_x[2][index] = b[2].x;
    data->tri1_y[0][index] = b[0].y;
    data->tri1_y[1][index] = b[1].y;
    data->tri1_y[2][index] = b[2].y;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_triangle_triangle);
}

//-----------------------------------------------------------------------------
void simdcol_segment_aabb(struct simdcol_context* context, uint32_t user_data, vec2 p0, vec2 p1, aabb box)
{
    assert(context->state == state_idle);
    assert(context->segment_aabb->num_items < BATCH_SIZE);

    struct segment_aabb_data* data = context->segment_aabb;
    uint32_t index = data->num_items++;

    data->p0_x[index] = p0.x;
    data->p1_x[index] = p1.x;
    data->p0_y[index] = p0.y;
    data->p1_y[index] = p1.y;
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_segment_aabb);
}

//-----------------------------------------------------------------------------
void simdcol_segment_disc(struct simdcol_context* context, uint32_t user_data, vec2 p0, vec2 p1, vec2 center, float radius)
{
    assert(context->state == state_idle);
    assert(context->segment_disc->num_items < BATCH_SIZE);

    struct segment_disc_data* data = context->segment_disc;
    uint32_t index = data->num_items++;

    data->p0_x[index] = p0.x;
    data->p1_x[index] = p1.x;
    data->p0_y[index] = p0.y;
    data->p1_y[index] = p1.y;
    data->center_x[index] = center.x;
    data->center_y[index] = center.y;
    data->sq_radius[index] = radius * radius;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_segment_disc);
}

//-----------------------------------------------------------------------------
void simdcol_triangle_disc(struct simdcol_context* context, uint32_t user_data, vec2 v0, vec2 v1, vec2 v2, vec2 disc_center, float disc_radius)
{
    assert(context->state == state_idle);
    assert(context->triangle_disc->num_items < BATCH_SIZE);

    struct triangle_disc_data* data = context->triangle_disc;
    uint32_t index = data->num_items++;

    data->v_x[0][index] = v0.x;
    data->v_x[1][index] = v1.x;
    data->v_x[2][index] = v2.x;
    data->v_y[0][index] = v0.y;
    data->v_y[1][index] = v1.y;
    data->v_y[2][index] = v2.y;
    data->center_x[index] = disc_center.x;
    data->center_y[index] = disc_center.y;
    data->sq_radius[index] = disc_radius * disc_radius;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_triangle_disc);
}

//-----------------------------------------------------------------------------
void simdcol_point_triangle(struct simdcol_context* context, uint32_t user_data, vec2 point, vec2 v0, vec2 v1, vec2 v2)
{
    assert(context->state == state_idle);
    assert(context->point_triangle->num_items < BATCH_SIZE);

    struct point_triangle_data* data = context->point_triangle;
    uint32_t index = data->num_items++;

    data->v_x[0][index] = v0.x;
    data->v_x[1][index] = v1.x;
    data->v_x[2][index] = v2.x;
    data->v_y[0][index] = v0.y;
    data->v_y[1][index] = v1.y;
    data->v_y[2][index] = v2.y;
    data->p_x[index] = point.x;
    data->p_y[index] = point.y;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_point_triangle);
}

//-----------------------------------------------------------------------------
void process_aabb_disc(struct simdcol_context* context)
{
    struct aabb_disc_data* data = context->aabb_disc;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 aabb_min = simd_vec2_load(data->aabb_min_x, data->aabb_min_y, offset);
        simd_vec2 aabb_max = simd_vec2_load(data->aabb_max_x, data->aabb_max_y,  offset);
        simd_vec2 center = simd_vec2_load(data->center_x, data->center_y, offset);
        simd_vec2 nearest = simd_vec2_clamp(center, aabb_min, aabb_max);
        simd_vec2 delta = simd_vec2_sub(nearest, center);
        simd_vector sq_distance = simd_vec2_sq_length(delta);
        simd_vector sq_radius = simd_load(data->sq_radius + offset);
        simd_vector result = simd_cmp_lt(sq_distance, sq_radius);

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if (bitfield&(1<<i) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
void process_aabb_circle(struct simdcol_context* context)
{
    struct aabb_circle_data* data = context->aabb_circle;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        // first check the aabb is in outter disc 
        simd_vec2 aabb_min = simd_vec2_load(data->aabb_min_x, data->aabb_min_y, offset);
        simd_vec2 aabb_max = simd_vec2_load(data->aabb_max_x, data->aabb_max_y,  offset);
        simd_vec2 center = simd_vec2_load(data->center_x, data->center_y, offset);
        simd_vec2 nearest = simd_vec2_clamp(center, aabb_min, aabb_max);
        simd_vec2 delta = simd_vec2_sub(nearest, center);
        simd_vector sq_distance = simd_vec2_sq_length(delta);
        simd_vector sq_outter_radius = simd_load(data->sq_outter_radius + offset);
        simd_vector sq_inner_radius = simd_load(data->sq_inner_radius + offset);
        simd_vector result = simd_cmp_lt(sq_distance, sq_outter_radius);
        
        // check the furthest corner is in inner disc
        simd_vec2 candidate0 = simd_vec2_abs(simd_vec2_sub(center, aabb_min));
        simd_vec2 candidate1 = simd_vec2_abs(simd_vec2_sub(center, aabb_max));
        simd_vec2 furthest = simd_vec2_max(candidate0, candidate1);
        result = simd_and(result, simd_cmp_gt(simd_vec2_sq_length(furthest), sq_inner_radius));

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if (bitfield&(1<<i) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector aabb_segment_test(simd_vec2 aabb_min, simd_vec2 aabb_max, simd_vec2 p0, simd_vec2 p1)
{
    simd_vector inv_dir_x = simd_rcp(simd_sub(p1.x, p0.x));
    simd_vector t0_x = simd_mul(inv_dir_x, simd_sub(aabb_min.x, p0.x));
    simd_vector t1_x = simd_mul(inv_dir_x, simd_sub(aabb_max.x, p0.x));
    simd_vector tmin_x = simd_min(t0_x, t1_x);
    simd_vector tmax_x = simd_max(t0_x, t1_x);

    simd_vector inv_dir_y = simd_rcp(simd_sub(p1.y, p0.y));
    simd_vector t0_y = simd_mul(inv_dir_y, simd_sub(aabb_min.y, p0.y));
    simd_vector t1_y = simd_mul(inv_dir_y, simd_sub(aabb_max.y, p0.y));
    simd_vector tmin_y = simd_min(t0_y, t1_y);
    simd_vector tmax_y = simd_max(t0_y, t1_y);
    
    simd_vector result = simd_cmp_le( simd_max(tmin_x, tmin_y), simd_min(tmax_x, tmax_y));
    result = simd_and(result, simd_cmp_ge(tmax_x, simd_splat_zero()));
    result = simd_and(result, simd_cmp_le(tmin_x, simd_splat(1.f)));
    result = simd_and(result, simd_cmp_ge(tmax_y, simd_splat_zero()));
    result = simd_and(result, simd_cmp_le(tmin_y, simd_splat(1.f)));
    return result;
}

//-----------------------------------------------------------------------------
void process_aabb_arc(struct simdcol_context* context)
{
    struct aabb_arc_data* data = context->aabb_arc;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        // first check the aabb is in outter disc 
        simd_vec2 aabb_min = simd_vec2_load(data->aabb_min_x, data->aabb_min_y, offset);
        simd_vec2 aabb_max = simd_vec2_load(data->aabb_max_x, data->aabb_max_y,  offset);
        simd_vec2 center = simd_vec2_load(data->center_x, data->center_y, offset);
        simd_vec2 nearest = simd_vec2_clamp(center, aabb_min, aabb_max);
        simd_vec2 delta = simd_vec2_sub(nearest, center);
        simd_vector sq_distance = simd_vec2_sq_length(delta);
        simd_vector outter_radius = simd_load(data->outter_radius + offset);
        simd_vector sq_outter_radius = simd_mul(outter_radius, outter_radius);
        simd_vector sq_inner_radius = simd_load(data->sq_inner_radius + offset);
        simd_vector result = simd_cmp_lt(sq_distance, sq_outter_radius);
        
        // check the furthest corner is in inner disc
        simd_vec2 candidate0 = simd_vec2_abs(simd_vec2_sub(center, aabb_min));
        simd_vec2 candidate1 = simd_vec2_abs(simd_vec2_sub(center, aabb_max));
        simd_vec2 furthest = simd_vec2_max(candidate0, candidate1);
        result = simd_and(result, simd_cmp_gt(simd_vec2_sq_length(furthest), sq_inner_radius));

        // check if the aabb intersects with the pie part of the disc
        // 1. test if any vertices of the aabb is in the pie
        simd_vector aperture = simd_load(data->aperture + offset);
        simd_vector orientation = simd_load(data->orientation + offset);
        simd_vec2 pie_direction = {.x = simd_cos(orientation), .y = simd_sin(orientation)};
        simd_vec2 to_vertex = simd_vec2_normalize(simd_vec2_sub(aabb_min, center));
        simd_vector angle = simd_acos(simd_vec2_dot(pie_direction, to_vertex));
        simd_vector pie_result = simd_cmp_le(angle, aperture);

        to_vertex = simd_vec2_normalize(simd_vec2_sub(aabb_max, center));
        angle = simd_acos(simd_vec2_dot(pie_direction, to_vertex));
        pie_result = simd_or(pie_result, simd_cmp_le(angle, aperture));

        to_vertex = simd_vec2_normalize(simd_vec2_sub((simd_vec2) {aabb_min.x, aabb_max.y}, center));
        angle = simd_acos(simd_vec2_dot(pie_direction, to_vertex));
        pie_result = simd_or(pie_result, simd_cmp_le(angle, aperture));

        to_vertex = simd_vec2_normalize(simd_vec2_sub((simd_vec2) {aabb_max.x, aabb_min.y}, center));
        angle = simd_acos(simd_vec2_dot(pie_direction, to_vertex));
        pie_result = simd_or(pie_result, simd_cmp_le(angle, aperture));

        // 2. test is the edges of the pie intersect with the aabb
        simd_vector edge_orientation = simd_sub(orientation, aperture);
        simd_vec2 edge = {.x = simd_cos(edge_orientation), .y = simd_sin(edge_orientation)};
        edge = simd_vec2_scale(edge, outter_radius);
        pie_result = simd_or(pie_result, aabb_segment_test(aabb_min, aabb_max, center, simd_vec2_add(center, edge)));

        edge_orientation = simd_add(orientation, aperture);
        edge = (simd_vec2) {.x = simd_cos(edge_orientation), .y = simd_sin(edge_orientation)};
        edge = simd_vec2_scale(edge, outter_radius);
        pie_result = simd_or(pie_result, aabb_segment_test(aabb_min, aabb_max, center, simd_vec2_add(center, edge)));

        result = simd_and(result, pie_result);

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if (bitfield&(1<<i) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}


//-----------------------------------------------------------------------------
static inline simd_vector all_greater3(simd_vector reference, simd_vector value0, simd_vector value1, simd_vector value2)
{
    simd_vector result = simd_cmp_gt(value0, reference);
    result = simd_and(result, simd_cmp_gt(value1, reference));
    return simd_and(result, simd_cmp_gt(value2, reference));
}

//-----------------------------------------------------------------------------
static inline simd_vector all_greater4(simd_vector reference, simd_vector value0, simd_vector value1, simd_vector value2, simd_vector value3)
{
    return simd_and(all_greater3(reference, value0, value1, value2), simd_cmp_gt(value3, reference));
}

//-----------------------------------------------------------------------------
static inline simd_vector all_less3(simd_vector reference, simd_vector value0, simd_vector value1, simd_vector value2)
{
    simd_vector result = simd_cmp_lt(value0, reference);
    result = simd_and(result, simd_cmp_lt(value1, reference));
    return simd_and(result, simd_cmp_lt(value2, reference));
}

//-----------------------------------------------------------------------------
static inline simd_vector all_less4(simd_vector reference, simd_vector value0, simd_vector value1, simd_vector value2, simd_vector value3)
{
    return simd_and(all_less3(reference, value0, value1, value2), simd_cmp_lt(value3, reference));
}

//-----------------------------------------------------------------------------
static inline simd_vector axis_aabb(simd_vector a, simd_vector b, simd_vector c, simd_vector extent,
                                    simd_vec2 aabb_min, simd_vec2 aabb_max)
{
    simd_vector distance_0 = simd_fmad(a, aabb_min.x, simd_fmad(b, aabb_min.y, c));
    simd_vector distance_1 = simd_fmad(a, aabb_min.x, simd_fmad(b, aabb_max.y, c));
    simd_vector distance_2 = simd_fmad(a, aabb_max.x, simd_fmad(b, aabb_min.y, c));
    simd_vector distance_3 = simd_fmad(a, aabb_max.x, simd_fmad(b, aabb_max.y, c));
    simd_vector result_gt = all_greater4(extent, distance_0, distance_1, distance_2, distance_3);
    simd_vector result_lt = all_less4(simd_neg(extent), distance_0, distance_1, distance_2, distance_3);
    return simd_or(result_gt, result_lt);
}

//-----------------------------------------------------------------------------
// Separating Axis Theorem
void process_aabb_obb(struct simdcol_context* context)
{
    struct aabb_obb_data* data = context->aabb_obb;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        // load everything (compiler will optimize simd register management)
        simd_vector j_a = simd_load(data->j_a + offset);
        simd_vector j_b = simd_load(data->j_b + offset);
        simd_vector j_c = simd_load(data->j_c + offset);
        simd_vector i_a = simd_neg(j_b);
        simd_vector i_b = j_a;
        simd_vector i_c = simd_load(data->i_c + offset);
        simd_vector extent_j = simd_load(data->extent_j + offset);
        simd_vector extent_i = simd_load(data->extent_i + offset);
        simd_vec2 aabb_min = simd_vec2_load(data->aabb_min_x, data->aabb_min_y, offset);
        simd_vec2 aabb_max = simd_vec2_load(data->aabb_max_x, data->aabb_max_y, offset);

        // compute obb vertices position
        simd_vector minus_j_c = simd_neg(j_c);
        simd_vector minus_i_c = simd_neg(i_c);
        simd_vector obb_v0_x = simd_fmad(j_a, simd_add(minus_j_c, extent_j), simd_mul(i_a, simd_add(minus_i_c, extent_i)));
        simd_vector obb_v1_x = simd_fmad(j_a, simd_sub(minus_j_c, extent_j), simd_mul(i_a, simd_add(minus_i_c, extent_i)));
        simd_vector obb_v2_x = simd_fmad(j_a, simd_add(minus_j_c, extent_j), simd_mul(i_a, simd_sub(minus_i_c, extent_i)));
        simd_vector obb_v3_x = simd_fmad(j_a, simd_sub(minus_j_c, extent_j), simd_mul(i_a, simd_sub(minus_i_c, extent_i)));
        simd_vector obb_v0_y = simd_fmad(j_b, simd_add(minus_j_c, extent_j), simd_mul(i_b, simd_add(minus_i_c, extent_i)));
        simd_vector obb_v1_y = simd_fmad(j_b, simd_sub(minus_j_c, extent_j), simd_mul(i_b, simd_add(minus_i_c, extent_i)));
        simd_vector obb_v2_y = simd_fmad(j_b, simd_add(minus_j_c, extent_j), simd_mul(i_b, simd_sub(minus_i_c, extent_i)));
        simd_vector obb_v3_y = simd_fmad(j_b, simd_sub(minus_j_c, extent_j), simd_mul(i_b, simd_sub(minus_i_c, extent_i)));

        // SAT : aabb axis
        simd_vector result = all_less4(aabb_min.x, obb_v0_x, obb_v1_x, obb_v2_x, obb_v3_x);
        result = simd_or(result, all_greater4(aabb_max.x, obb_v0_x, obb_v1_x, obb_v2_x, obb_v3_x));
        result = simd_or(result, all_less4(aabb_min.y, obb_v0_y, obb_v1_y, obb_v2_y, obb_v3_y));
        result = simd_or(result, all_greater4(aabb_max.y, obb_v0_y, obb_v1_y, obb_v2_y, obb_v3_y));

        // SAT : obb axis
        result = simd_or(result, axis_aabb(j_a, j_b, j_c, extent_j, aabb_min, aabb_max));
        result = simd_or(result, axis_aabb(i_a, i_b, i_c, extent_i, aabb_min, aabb_max)); 
        
        int bitfield = simd_get_mask(result);

        for(uint32_t i=0; i<simd_vector_width; ++i)
            if ((bitfield&(1<<i)) == 0 && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector edge_triangle(simd_vec2 a0, simd_vec2 a1, simd_vec2 a2, 
                                        simd_vec2 b0, simd_vec2 b1, simd_vec2 b2)
{
    simd_vector edge_a = simd_sub(a0.y, a1.y);
    simd_vector edge_b = simd_sub(a1.x, a0.x);
    simd_vector edge_c = simd_sub(simd_mul(a0.x, a1.y), simd_mul(a0.y, a1.x));

    simd_vector d0 = simd_fmad(edge_a, b0.x, simd_fmad(edge_b, b0.y, edge_c));
    simd_vector d1 = simd_fmad(edge_a, b1.x, simd_fmad(edge_b, b1.y, edge_c));
    simd_vector d2 = simd_fmad(edge_a, b2.x, simd_fmad(edge_b, b2.y, edge_c));
    simd_vector other = simd_fmad(edge_a, a2.x, simd_fmad(edge_b, a2.y, edge_c));

    simd_vector zero = simd_splat_zero();

    simd_vector r0 = all_less3(zero, d0, d1, d2); simd_cmp_lt(d0, zero);
    r0 = simd_and(r0, simd_cmp_gt(other, zero));

    simd_vector r1 = all_greater3(zero, d0, d1, d2);
    r1 = simd_and(r1, simd_cmp_lt(other, zero));

    return simd_or(r0, r1);
}

//-----------------------------------------------------------------------------
// Separating Axis Theorem
void process_triangle_triangle(struct simdcol_context* context)
{
    struct triangle_triangle_data* data = context->triangle_triangle;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 a0 = simd_vec2_load(data->tri0_x[0], data->tri0_y[0], offset);
        simd_vec2 a1 = simd_vec2_load(data->tri0_x[1], data->tri0_y[1], offset);
        simd_vec2 a2 = simd_vec2_load(data->tri0_x[2], data->tri0_y[2], offset);
        simd_vec2 b0 = simd_vec2_load(data->tri1_x[0], data->tri1_y[0], offset);
        simd_vec2 b1 = simd_vec2_load(data->tri1_x[1], data->tri1_y[1], offset);
        simd_vec2 b2 = simd_vec2_load(data->tri1_x[2], data->tri1_y[2], offset);

        simd_vector result = edge_triangle(a0, a1, a2, b0, b1, b2);
        result = simd_or(result, edge_triangle(a1, a2, a0, b0, b1, b2));
        result = simd_or(result, edge_triangle(a2, a0, a1, b0, b1, b2));
        result = simd_or(result, edge_triangle(b0, b1, b2, a0, a1, a2));
        result = simd_or(result, edge_triangle(b1, b2, b0, a0, a1, a2));
        result = simd_or(result, edge_triangle(b2, b0, b1, a0, a1, a2));

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if ((bitfield&(1<<i)) == 0 && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector edge_triangle_aabb(simd_vec2 a0, simd_vec2 a1, simd_vec2 a2, simd_vec2 min, simd_vec2 max)
{
    simd_vector edge_a = simd_sub(a0.y, a1.y);
    simd_vector edge_b = simd_sub(a1.x, a0.x);
    simd_vector edge_c = simd_sub(simd_mul(a0.x, a1.y), simd_mul(a0.y, a1.x));

    simd_vector d0 = simd_fmad(edge_a, min.x, simd_fmad(edge_b, min.y, edge_c));
    simd_vector d1 = simd_fmad(edge_a, min.x, simd_fmad(edge_b, max.y, edge_c));
    simd_vector d2 = simd_fmad(edge_a, max.x, simd_fmad(edge_b, min.y, edge_c));
    simd_vector d3 = simd_fmad(edge_a, max.x, simd_fmad(edge_b, max.y, edge_c));
    simd_vector other = simd_fmad(edge_a, a2.x, simd_fmad(edge_b, a2.y, edge_c));

    simd_vector zero = simd_splat_zero();

    simd_vector r0 = all_less4(zero, d0, d1, d2, d3);
    r0 = simd_and(r0, simd_cmp_gt(other, zero));

    simd_vector r1 = all_greater4(zero, d0, d1, d2, d3);
    r1 = simd_and(r1, simd_cmp_lt(other, zero));

    return simd_or(r0, r1);
}

//-----------------------------------------------------------------------------
// Separating Axis Theorem
void process_aabb_triangle(struct simdcol_context* context)
{
    struct aabb_triangle_data* data = context->aabb_triangle;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 aabb_min = simd_vec2_load(data->aabb_min_x, data->aabb_min_y, offset);
        simd_vec2 aabb_max = simd_vec2_load(data->aabb_max_x, data->aabb_max_y, offset);
        simd_vec2 v0 = simd_vec2_load(data->v_x[0], data->v_y[0], offset);
        simd_vec2 v1 = simd_vec2_load(data->v_x[1], data->v_y[1], offset);
        simd_vec2 v2 = simd_vec2_load(data->v_x[2], data->v_y[2], offset);

        // SAT : aabb axis
        simd_vector result = all_less3(aabb_min.x, v0.x, v1.x, v2.x);
        result = simd_or(result, all_greater3(aabb_max.x, v0.x, v1.x, v2.x)); 
        result = simd_or(result, all_less3(aabb_min.y, v0.y, v1.y, v2.y));
        result = simd_or(result, all_greater3(aabb_max.y, v0.y, v1.y, v2.y));

        // SAT : triangle axis
        result = simd_or(result, edge_triangle_aabb(v0, v1, v2, aabb_min, aabb_max));
        result = simd_or(result, edge_triangle_aabb(v1, v2, v0, aabb_min, aabb_max));
        result = simd_or(result, edge_triangle_aabb(v2, v0, v1, aabb_min, aabb_max));

        int bitfield = simd_get_mask(result);

        for(uint32_t i=0; i<simd_vector_width; ++i)
            if ((bitfield&(1<<i)) == 0 && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
void process_segment_aabb(struct simdcol_context* context)
{
    struct segment_aabb_data* data = context->segment_aabb;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 aabb_min = simd_vec2_load(data->aabb_min_x, data->aabb_min_y, offset);
        simd_vec2 aabb_max = simd_vec2_load(data->aabb_max_x, data->aabb_max_y, offset);
        simd_vec2 p0 = simd_vec2_load(data->p0_x, data->p0_y, offset);
        simd_vec2 p1 = simd_vec2_load(data->p1_x, data->p1_y, offset);
        simd_vector result = aabb_segment_test(aabb_min, aabb_max, p0, p1);

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if ((bitfield&(1<<i)) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector sq_distance_to_segment(simd_vec2 point, simd_vec2 a, simd_vec2 b)
{
    simd_vec2 pa = simd_vec2_sub(point, a);
    simd_vec2 ba = simd_vec2_sub(b, a);
    simd_vector h = simd_saturate(simd_div(simd_vec2_dot(pa, ba), simd_vec2_sq_length(ba)));
    simd_vec2 delta = simd_vec2_sub(pa, simd_vec2_mul(ba, (simd_vec2) {h, h}));
    return simd_vec2_sq_length(delta);
}

//-----------------------------------------------------------------------------
// simply compute squared distance between center and the segment and compare it to squared radius
void process_segment_disc(struct simdcol_context* context)
{
    struct segment_disc_data* data = context->segment_disc;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 p0 = simd_vec2_load(data->p0_x, data->p0_y, offset);
        simd_vec2 p1 = simd_vec2_load(data->p1_x, data->p1_y, offset);
        simd_vec2 center = simd_vec2_load(data->center_x, data->center_y, offset);
        simd_vector sq_radius = simd_load(data->sq_radius + offset);
        simd_vector result = simd_cmp_le(sq_distance_to_segment(center, p0, p1), sq_radius);

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if ((bitfield&(1<<i)) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector edge_sign(simd_vec2 p, simd_vec2 e0, simd_vec2 e1)
{
    return simd_sub(simd_mul(simd_sub(p.x, e1.x), simd_sub(e0.y, e1.y)),
                    simd_mul(simd_sub(e0.x, e1.x), simd_sub(p.y, e1.y)));
}

//-----------------------------------------------------------------------------
static simd_vector point_in_triangle(simd_vec2 p, simd_vec2 v0, simd_vec2 v1, simd_vec2 v2)
{
    simd_vector d0 = edge_sign(p, v0, v1);
    simd_vector d1 = edge_sign(p, v1, v2);
    simd_vector d2 = edge_sign(p, v2, v0);

    simd_vector all_positive = simd_cmp_ge(d0, simd_splat_zero());
    all_positive = simd_and(all_positive, simd_cmp_ge(d1, simd_splat_zero()));
    all_positive = simd_and(all_positive, simd_cmp_ge(d2, simd_splat_zero()));

    simd_vector all_negative = simd_cmp_lt(d0, simd_splat_zero());
    all_negative = simd_and(all_negative, simd_cmp_lt(d1, simd_splat_zero()));
    all_negative = simd_and(all_negative, simd_cmp_lt(d2, simd_splat_zero()));

    return simd_or(all_positive, all_negative);
}

//-----------------------------------------------------------------------------
void process_point_triangle(struct simdcol_context* context)
{
    struct point_triangle_data* data = context->point_triangle;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 v0 = simd_vec2_load(data->v_x[0], data->v_y[0], offset);
        simd_vec2 v1 = simd_vec2_load(data->v_x[1], data->v_y[1], offset);
        simd_vec2 v2 = simd_vec2_load(data->v_x[2], data->v_y[2], offset);
        simd_vec2 p = simd_vec2_load(data->p_x, data->p_y, offset);

        simd_vector result = point_in_triangle(p, v0, v1, v2);
        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if ((bitfield&(1<<i)) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
// based on signed distance field https://iquilezles.org/articles/distfunctions2d/
void process_triangle_disc(struct simdcol_context* context)
{
    struct triangle_disc_data* data = context->triangle_disc;

    uint32_t num_vec = simd_num_vec(data->num_items);
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 p0 = simd_vec2_load(data->v_x[0], data->v_y[0], offset);
        simd_vec2 p1 = simd_vec2_load(data->v_x[1], data->v_y[1], offset);
        simd_vec2 p2 = simd_vec2_load(data->v_x[2], data->v_y[2], offset);
        simd_vec2 center = simd_vec2_load(data->center_x, data->center_y, offset);
        simd_vector sq_radius = simd_load(data->sq_radius + offset);

        simd_vec2 e0 = simd_vec2_sub(p1, p0); simd_vec2 e1 = simd_vec2_sub(p2, p1); simd_vec2 e2 = simd_vec2_sub(p0, p2);
        simd_vec2 v0 = simd_vec2_sub(center, p0); simd_vec2 v1 = simd_vec2_sub(center, p1); simd_vec2 v2 = simd_vec2_sub(center, p2);

        simd_vec2 pq0 = simd_vec2_sub(v0,  simd_vec2_scale(e0, simd_saturate(simd_div(simd_vec2_dot(v0, e0), simd_vec2_sq_length(e0)))));
        simd_vec2 pq1 = simd_vec2_sub(v1,  simd_vec2_scale(e1, simd_saturate(simd_div(simd_vec2_dot(v1, e1), simd_vec2_sq_length(e1)))));
        simd_vec2 pq2 = simd_vec2_sub(v2,  simd_vec2_scale(e2, simd_saturate(simd_div(simd_vec2_dot(v2, e2), simd_vec2_sq_length(e2)))));

        simd_vector s = simd_sign(simd_fmad(e0.x, e2.y, simd_mul(simd_neg(e0.y), e2.x)));
        simd_vector squared_distance = simd_min(simd_vec2_sq_length(pq0), simd_min(simd_vec2_sq_length(pq1), simd_vec2_sq_length(pq2)));
        simd_vector sign =    simd_mul(s, simd_fmad(v0.x, e0.y, simd_mul(simd_neg(v0.y), e0.x)));
        sign = simd_min(sign, simd_mul(s, simd_fmad(v1.x, e1.y, simd_mul(simd_neg(v1.y), e1.x))));
        sign = simd_min(sign, simd_mul(s, simd_fmad(v2.x, e2.y, simd_mul(simd_neg(v2.y), e2.x))));

        squared_distance = simd_mul(simd_neg(squared_distance), simd_sign(sign));
        simd_vector result = simd_cmp_lt(squared_distance, sq_radius);
        
        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width; ++i)
            if ((bitfield&(1<<i)) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
void simdcol_flush(struct simdcol_context* context, enum flush_hint hint)
{
    assert(context->state == state_idle);
    context->state = state_processing;

    if (hint == flush_aabb_disc || hint == flush_all)
        process_aabb_disc(context);

    if (hint == flush_aabb_circle || hint == flush_all)
        process_aabb_circle(context);

    if (hint == flush_aabb_arc || hint == flush_all)
        process_aabb_arc(context);

    if (hint == flush_aabb_triangle || hint == flush_all)
        process_aabb_triangle(context);

    if (hint == flush_aabb_obb || hint == flush_all)
        process_aabb_obb(context);

    if (hint == flush_triangle_triangle || hint == flush_all)
        process_triangle_triangle(context);

    if (hint == flush_segment_aabb || hint == flush_all)
        process_segment_aabb(context);

    if (hint == flush_segment_disc || hint == flush_all)
        process_segment_disc(context);

    if (hint == flush_triangle_disc || hint == flush_all)
        process_triangle_disc(context);

    if (hint == flush_point_triangle || hint == flush_all)
        process_point_triangle(context);

    context->state = state_idle;
}

//-----------------------------------------------------------------------------
void simdcol_terminate(struct simdcol_context* context)
{
    simd_aligned_free(context->aabb_triangle);
    simd_aligned_free(context->aabb_obb);
    simd_aligned_free(context->aabb_disc);
    simd_aligned_free(context->aabb_circle);
    simd_aligned_free(context->aabb_arc);
    simd_aligned_free(context->triangle_triangle);
    simd_aligned_free(context->segment_aabb);
    simd_aligned_free(context->segment_disc);
    simd_aligned_free(context->triangle_disc);
    simd_aligned_free(context->point_triangle);
    free(context);
}
