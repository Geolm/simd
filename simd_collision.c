#include "simd_collision.h"
#include "simd_vec2.h"

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
struct aabb_circle_data
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
struct segment_circle_data
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
struct triangle_circle_data
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
    struct aabb_circle_data* aabb_circle;
    struct triangle_triangle_data* triangle_triangle;
    struct segment_aabb_data* segment_aabb;
    struct segment_circle_data* segment_circle;
    struct triangle_circle_data* triangle_circle;

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
    context->aabb_circle = (struct aabb_circle_data*) simd_aligned_alloc(sizeof(struct aabb_circle_data));
    context->triangle_triangle = (struct triangle_triangle_data*) simd_aligned_alloc(sizeof(struct triangle_triangle_data));
    context->segment_aabb = (struct segment_aabb_data*) simd_aligned_alloc(sizeof(struct segment_aabb_data));
    context->segment_circle = (struct segment_circle_data*) simd_aligned_alloc(sizeof(struct segment_circle_data));
    context->triangle_circle = (struct triangle_circle_data*) simd_aligned_alloc(sizeof(struct triangle_circle_data));

    context->aabb_triangle->num_items = 0;
    context->aabb_obb->num_items = 0;
    context->aabb_circle->num_items = 0;
    context->triangle_triangle->num_items = 0;
    context->segment_aabb->num_items = 0;
    context->segment_circle->num_items = 0;
    context->triangle_circle->num_items = 0;

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
void simdcol_aabb_circle(struct simdcol_context* context, uint32_t user_data, aabb box, circle c)
{
    assert(context->aabb_circle->num_items < BATCH_SIZE);
    assert(context->state == state_idle);

    struct aabb_circle_data* data = context->aabb_circle;
    uint32_t index = data->num_items++;

    // copy data in SoA
    data->user_data[index] = user_data;
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->center_x[index] = c.center.x;
    data->center_y[index] = c.center.y;
    data->sq_radius[index] = c.radius * c.radius;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_aabb_circle);
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
void simdcol_segment_aabb(struct simdcol_context* context, uint32_t user_data, segment line, aabb box)
{
    assert(context->state == state_idle);
    assert(context->segment_aabb->num_items < BATCH_SIZE);

    struct segment_aabb_data* data = context->segment_aabb;
    uint32_t index = data->num_items++;

    data->p0_x[index] = line.p0.x;
    data->p1_x[index] = line.p1.x;
    data->p0_y[index] = line.p0.y;
    data->p1_y[index] = line.p1.y;
    data->aabb_min_x[index] = box.min.x;
    data->aabb_min_y[index] = box.min.y;
    data->aabb_max_x[index] = box.max.x;
    data->aabb_max_y[index] = box.max.y;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_segment_aabb);
}

//-----------------------------------------------------------------------------
void simdcol_segment_circle(struct simdcol_context* context, uint32_t user_data, segment line, circle c)
{
    assert(context->state == state_idle);
    assert(context->segment_circle->num_items < BATCH_SIZE);

    struct segment_circle_data* data = context->segment_circle;
    uint32_t index = data->num_items++;

    data->p0_x[index] = line.p0.x;
    data->p1_x[index] = line.p1.x;
    data->p0_y[index] = line.p0.y;
    data->p1_y[index] = line.p1.y;
    data->center_x[index] = c.center.x;
    data->center_y[index] = c.center.y;
    data->sq_radius[index] = c.radius * c.radius;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_segment_circle);
}

//-----------------------------------------------------------------------------
void simdcol_triangle_circle(struct simdcol_context* context, uint32_t user_data, vec2 v0, vec2 v1, vec2 v2, circle c)
{
    assert(context->state == state_idle);
    assert(context->triangle_circle->num_items < BATCH_SIZE);

    struct triangle_circle_data* data = context->triangle_circle;
    uint32_t index = data->num_items++;

    data->v_x[0][index] = v0.x;
    data->v_x[1][index] = v1.x;
    data->v_x[2][index] = v2.x;
    data->v_y[0][index] = v0.y;
    data->v_y[1][index] = v1.y;
    data->v_y[2][index] = v2.y;
    data->center_x[index] = c.center.x;
    data->center_y[index] = c.center.y;
    data->sq_radius[index] = c.radius * c.radius;
    data->user_data[index] = user_data;

    if (data->num_items == BATCH_SIZE)
        simdcol_flush(context, flush_triangle_circle);
}

//-----------------------------------------------------------------------------
void process_aabb_circle(struct simdcol_context* context)
{
    struct aabb_circle_data* data = context->aabb_circle;

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
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

    if (data->num_items == 0)
        return;
    
    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
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

        for(uint32_t i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
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

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
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
        for(uint32_t i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
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

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
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

        for(uint32_t i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
            if ((bitfield&(1<<i)) == 0 && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector slab_test(simd_vector slab_min, simd_vector slab_max, simd_vector segment_start, simd_vector segment_end)
{
    simd_vector inv_dir = simd_rcp(simd_sub(segment_end, segment_start));
    simd_vector t0 = simd_mul(inv_dir, simd_sub(slab_min, segment_start));
    simd_vector t1 = simd_mul(inv_dir, simd_sub(slab_max, segment_start));
    simd_vector enter = simd_min(t0, t1);
    simd_vector exit = simd_max(t0, t1);
    return simd_and(simd_cmp_lt(enter, simd_splat(1.f)), simd_cmp_gt(exit, simd_splat_zero()));
}

//-----------------------------------------------------------------------------
void process_segment_aabb(struct simdcol_context* context)
{
    struct segment_aabb_data* data = context->segment_aabb;

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 aabb_min = simd_vec2_load(data->aabb_min_x, data->aabb_min_y, offset);
        simd_vec2 aabb_max = simd_vec2_load(data->aabb_max_x, data->aabb_max_y, offset);
        simd_vec2 p0 = simd_vec2_load(data->p0_x, data->p0_y, offset);
        simd_vec2 p1 = simd_vec2_load(data->p1_x, data->p1_y, offset);
        simd_vector result = slab_test(aabb_min.x, aabb_max.x, p0.x, p1.x);
        result = simd_and(result, slab_test(aabb_min.y, aabb_max.y, p0.y, p1.y));

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
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
void process_segment_circle(struct simdcol_context* context)
{
    struct segment_circle_data* data = context->segment_circle;

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 p0 = simd_vec2_load(data->p0_x, data->p0_y, offset);
        simd_vec2 p1 = simd_vec2_load(data->p1_x, data->p1_y, offset);
        simd_vec2 center = simd_vec2_load(data->center_x, data->center_y, offset);
        simd_vector sq_radius = simd_load(data->sq_radius + offset);
        simd_vector result = simd_cmp_le(sq_distance_to_segment(center, p0, p1), sq_radius);

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
            if ((bitfield&(1<<i)) && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }
    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector edge_sign(simd_vec2 p, simd_vec2 e0, simd_vec2 e1)
{
    return simd_sub(simd_mul(simd_sub(p.x, e1.x), simd_sub(e0.y, e1.y)), simd_mul(simd_sub(e0.x, e1.x), simd_mul(p.y, e1.y)));
}

//-----------------------------------------------------------------------------
static simd_vector point_in_triangle(simd_vec2 p, simd_vec2 v0, simd_vec2 v1, simd_vec2 v2)
{
    simd_vector d0 = edge_sign(p, v0, v1);
    simd_vector d1 = edge_sign(p, v1, v2);
    simd_vector d2 = edge_sign(p, v2, v0);

    simd_vector all_positive = simd_cmp_gt(d0, simd_splat_zero());
    all_positive = simd_and(all_positive, simd_cmp_gt(d1, simd_splat_zero()));
    all_positive = simd_and(all_positive, simd_cmp_gt(d2, simd_splat_zero()));

    simd_vector all_negative = simd_cmp_lt(d0, simd_splat_zero());
    all_negative = simd_and(all_negative, simd_cmp_lt(d1, simd_splat_zero()));
    all_negative = simd_and(all_negative, simd_cmp_lt(d2, simd_splat_zero()));

    return simd_or(all_positive, all_negative);
}

//-----------------------------------------------------------------------------
// 2 steps : 
//   - center of circle in the triangle
//   - triangle's edges intersection with circle
void process_triangle_circle(struct simdcol_context* context)
{
    struct triangle_circle_data* data = context->triangle_circle;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        uint32_t offset = vec_index * simd_vector_width;

        simd_vec2 v0 = simd_vec2_load(data->v_x[0], data->v_y[0], offset);
        simd_vec2 v1 = simd_vec2_load(data->v_x[1], data->v_y[1], offset);
        simd_vec2 v2 = simd_vec2_load(data->v_x[2], data->v_y[2], offset);
        simd_vec2 center = simd_vec2_load(data->center_x, data->center_y, offset);
        simd_vector sq_radius = simd_load(data->sq_radius + offset);

        simd_vector result = point_in_triangle(center, v0, v1, v2);
        result = simd_or(result, simd_cmp_le(sq_distance_to_segment(center, v0, v1), sq_radius));
        result = simd_or(result, simd_cmp_le(sq_distance_to_segment(center, v1, v2), sq_radius));
        result = simd_or(result, simd_cmp_le(sq_distance_to_segment(center, v2, v0), sq_radius));

        int bitfield = simd_get_mask(result);
        for(uint32_t i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
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

    if (hint == flush_aabb_circle || hint == flush_all)
        process_aabb_circle(context);

    if (hint == flush_aabb_triangle || hint == flush_all)
        process_aabb_triangle(context);

    if (hint == flush_aabb_obb || hint == flush_all)
        process_aabb_obb(context);

    if (hint == flush_triangle_triangle || hint == flush_all)
        process_triangle_triangle(context);

    if (hint == flush_segment_aabb || hint == flush_all)
        process_segment_aabb(context);

    if (hint == flush_segment_circle || hint == flush_all)
        process_segment_circle(context);

    if (hint == flush_triangle_circle || hint == flush_all)
        process_triangle_circle(context);

    context->state = state_idle;
}

//-----------------------------------------------------------------------------
void simdcol_terminate(struct simdcol_context* context)
{
    simd_aligned_free(context->aabb_triangle);
    simd_aligned_free(context->aabb_obb);
    simd_aligned_free(context->aabb_circle);
    simd_aligned_free(context->triangle_triangle);
    simd_aligned_free(context->segment_aabb);
    simd_aligned_free(context->segment_circle);
    simd_aligned_free(context->triangle_circle);
    free(context);
}
