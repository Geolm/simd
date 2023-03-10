#include "simd_collision.h"
#include "simd.h"

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

    void* user_context;
    simdcol_intersection_callback on_intersection;
    enum simdcol_state state;
};

//-----------------------------------------------------------------------------
struct simdcol_context* simdcol_init(void* user_context, simdcol_intersection_callback callback)
{
    struct simdcol_context* context = (struct simdcol_context*) malloc(sizeof(struct simdcol_context));

    context->aabb_triangle = (struct aabb_triangle_data*) simd_aligned_alloc(sizeof(struct aabb_triangle_data));
    context->aabb_obb = (struct aabb_obb_data*) simd_aligned_alloc(sizeof(struct aabb_obb_data));
    context->aabb_circle = (struct aabb_circle_data*) simd_aligned_alloc(sizeof(struct aabb_circle_data));
    context->triangle_triangle = (struct triangle_triangle_data*) simd_aligned_alloc(sizeof(struct triangle_triangle_data));
    context->segment_aabb = (struct segment_aabb_data*) simd_aligned_alloc(sizeof(struct segment_aabb_data));

    context->aabb_triangle->num_items = 0;
    context->aabb_obb->num_items = 0;
    context->aabb_circle->num_items = 0;
    context->triangle_triangle->num_items = 0;
    context->segment_aabb->num_items = 0;

    context->state = state_idle;
    context->on_intersection = callback;
    context->user_context = user_context;

    return context;
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
    unsigned int index = data->num_items++;

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
    unsigned int index = data->num_items++;

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
    unsigned int index = data->num_items++;

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
void process_aabb_circle(struct simdcol_context* context)
{
    struct aabb_circle_data* data = context->aabb_circle;

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        int offset = vec_index * simd_vector_width;

        simd_vector aabb_min_x = simd_load(data->aabb_min_x + offset);
        simd_vector aabb_min_y = simd_load(data->aabb_min_y + offset);
        simd_vector aabb_max_x = simd_load(data->aabb_max_x + offset);
        simd_vector aabb_max_y = simd_load(data->aabb_max_y + offset);
        simd_vector center_x = simd_load(data->center_x + offset);
        simd_vector center_y = simd_load(data->center_y + offset);
        simd_vector nearest_x = simd_clamp(center_x, aabb_min_x, aabb_max_x);
        simd_vector nearest_y = simd_clamp(center_y, aabb_min_y, aabb_max_y);
        simd_vector delta_x = simd_sub(nearest_x, center_x);
        simd_vector delta_y = simd_sub(nearest_y, center_y);
        simd_vector sq_distance = simd_fmad(delta_x, delta_x, simd_mul(delta_y, delta_y));
        simd_vector sq_radius = simd_load(data->sq_radius + offset);
        simd_vector result = simd_cmp_lt(sq_distance, sq_radius);

        int bitfield = simd_get_mask(result);

        for(int i=0; i<simd_vector_width; ++i)
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
                                    simd_vector aabb_min_x, simd_vector aabb_min_y,
                                    simd_vector aabb_max_x, simd_vector aabb_max_y)
{
    simd_vector distance_0 = simd_fmad(a, aabb_min_x, simd_fmad(b, aabb_min_y, c));
    simd_vector distance_1 = simd_fmad(a, aabb_min_x, simd_fmad(b, aabb_max_y, c));
    simd_vector distance_2 = simd_fmad(a, aabb_max_x, simd_fmad(b, aabb_min_y, c));
    simd_vector distance_3 = simd_fmad(a, aabb_max_x, simd_fmad(b, aabb_max_y, c));

    simd_vector result_gt = all_greater4(extent, distance_0, distance_1, distance_2, distance_3);
    simd_vector result_lt = all_less4(simd_neg(extent), distance_0, distance_1, distance_2, distance_3);

    return simd_or(result_gt, result_lt);
}

//-----------------------------------------------------------------------------
void process_aabb_obb(struct simdcol_context* context)
{
    struct aabb_obb_data* data = context->aabb_obb;

    if (data->num_items == 0)
        return;
    
    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        int offset = vec_index * simd_vector_width;

        // load everything (compiler will optimize simd register management)
        simd_vector j_a = simd_load(data->j_a + offset);
        simd_vector j_b = simd_load(data->j_b + offset);
        simd_vector j_c = simd_load(data->j_c + offset);
        simd_vector i_a = simd_neg(j_b);
        simd_vector i_b = j_a;
        simd_vector i_c = simd_load(data->i_c + offset);
        simd_vector extent_j = simd_load(data->extent_j + offset);
        simd_vector extent_i = simd_load(data->extent_i + offset);
        simd_vector aabb_min_x = simd_load(data->aabb_min_x + offset);
        simd_vector aabb_min_y = simd_load(data->aabb_min_y + offset);
        simd_vector aabb_max_x = simd_load(data->aabb_max_x + offset);
        simd_vector aabb_max_y = simd_load(data->aabb_max_y + offset);

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
        simd_vector result = all_less4(aabb_min_x, obb_v0_x, obb_v1_x, obb_v2_x, obb_v3_x);
        result = simd_or(result, all_greater4(aabb_max_x, obb_v0_x, obb_v1_x, obb_v2_x, obb_v3_x));
        result = simd_or(result, all_less4(aabb_min_y, obb_v0_y, obb_v1_y, obb_v2_y, obb_v3_y));
        result = simd_or(result, all_greater4(aabb_max_y, obb_v0_y, obb_v1_y, obb_v2_y, obb_v3_y));

        // SAT : obb axis
        result = simd_or(result, axis_aabb(j_a, j_b, j_c, extent_j, aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y));
        result = simd_or(result, axis_aabb(i_a, i_b, i_c, extent_i, aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y)); 
        
        int bitfield = simd_get_mask(result);

        for(int i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
            if ((bitfield&(1<<i)) == 0 && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }

    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector edge_triangle(simd_vector a0_x, simd_vector a0_y,
                                        simd_vector a1_x, simd_vector a1_y,
                                        simd_vector a2_x, simd_vector a2_y, 
                                        simd_vector b0_x, simd_vector b0_y,
                                        simd_vector b1_x, simd_vector b1_y,
                                        simd_vector b2_x, simd_vector b2_y)
{
    simd_vector edge_a = simd_sub(a0_y, a1_y);
    simd_vector edge_b = simd_sub(a1_x, a0_x);
    simd_vector edge_c = simd_sub(simd_mul(a0_x, a1_y), simd_mul(a0_y, a1_x));

    simd_vector d0 = simd_fmad(edge_a, b0_x, simd_fmad(edge_b, b0_y, edge_c));
    simd_vector d1 = simd_fmad(edge_a, b1_x, simd_fmad(edge_b, b1_y, edge_c));
    simd_vector d2 = simd_fmad(edge_a, b2_x, simd_fmad(edge_b, b2_y, edge_c));
    simd_vector other = simd_fmad(edge_a, a2_x, simd_fmad(edge_b, a2_y, edge_c));

    simd_vector zero = simd_splat_zero();

    simd_vector r0 = all_less3(zero, d0, d1, d2); simd_cmp_lt(d0, zero);
    r0 = simd_and(r0, simd_cmp_gt(other, zero));

    simd_vector r1 = all_greater3(zero, d0, d1, d2);
    r1 = simd_and(r1, simd_cmp_lt(other, zero));

    return simd_or(r0, r1);
}

//-----------------------------------------------------------------------------
void process_triangle_triangle(struct simdcol_context* context)
{
    struct triangle_triangle_data* data = context->triangle_triangle;

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        int offset = vec_index * simd_vector_width;

        simd_vector a0_x = simd_load(data->tri0_x[0] + offset);
        simd_vector a1_x = simd_load(data->tri0_x[1] + offset);
        simd_vector a2_x = simd_load(data->tri0_x[2] + offset);
        simd_vector a0_y = simd_load(data->tri0_y[0] + offset);
        simd_vector a1_y = simd_load(data->tri0_y[1] + offset);
        simd_vector a2_y = simd_load(data->tri0_y[2] + offset);
        simd_vector b0_x = simd_load(data->tri1_x[0] + offset);
        simd_vector b1_x = simd_load(data->tri1_x[1] + offset);
        simd_vector b2_x = simd_load(data->tri1_x[2] + offset);
        simd_vector b0_y = simd_load(data->tri1_y[0] + offset);
        simd_vector b1_y = simd_load(data->tri1_y[1] + offset);
        simd_vector b2_y = simd_load(data->tri1_y[2] + offset);

        simd_vector result = edge_triangle(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, b0_x, b0_y, b1_x, b1_y, b2_x, b2_y);
        result = simd_or(result, edge_triangle(a1_x, a1_y, a2_x, a2_y, a0_x, a0_y, b0_x, b0_y, b1_x, b1_y, b2_x, b2_y));
        result = simd_or(result, edge_triangle(a2_x, a2_y, a0_x, a0_y, a1_x, a1_y, b0_x, b0_y, b1_x, b1_y, b2_x, b2_y));
        result = simd_or(result, edge_triangle(b0_x, b0_y, b1_x, b1_y, b2_x, b2_y, a0_x, a0_y, a1_x, a1_y, a2_x, a2_y));
        result = simd_or(result, edge_triangle(b1_x, b1_y, b2_x, b2_y, b0_x, b0_y, a0_x, a0_y, a1_x, a1_y, a2_x, a2_y));
        result = simd_or(result, edge_triangle(b2_x, b2_y, b0_x, b0_y, b1_x, b1_y, a0_x, a0_y, a1_x, a1_y, a2_x, a2_y));

        int bitfield = simd_get_mask(result);

        for(int i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
            if ((bitfield&(1<<i)) == 0 && (offset + i) < data->num_items)
                context->on_intersection(context->user_context, data->user_data[offset + i]);
    }

    data->num_items = 0;
}

//-----------------------------------------------------------------------------
static inline simd_vector edge_triangle_aabb(simd_vector a0_x, simd_vector a0_y,
                                             simd_vector a1_x, simd_vector a1_y,
                                             simd_vector a2_x, simd_vector a2_y, 
                                             simd_vector min_x, simd_vector min_y,
                                             simd_vector max_x, simd_vector max_y)
{
    simd_vector edge_a = simd_sub(a0_y, a1_y);
    simd_vector edge_b = simd_sub(a1_x, a0_x);
    simd_vector edge_c = simd_sub(simd_mul(a0_x, a1_y), simd_mul(a0_y, a1_x));

    simd_vector d0 = simd_fmad(edge_a, min_x, simd_fmad(edge_b, min_y, edge_c));
    simd_vector d1 = simd_fmad(edge_a, min_x, simd_fmad(edge_b, max_y, edge_c));
    simd_vector d2 = simd_fmad(edge_a, max_x, simd_fmad(edge_b, min_y, edge_c));
    simd_vector d3 = simd_fmad(edge_a, max_x, simd_fmad(edge_b, max_y, edge_c));
    simd_vector other = simd_fmad(edge_a, a2_x, simd_fmad(edge_b, a2_y, edge_c));

    simd_vector zero = simd_splat_zero();

    simd_vector r0 = all_less4(zero, d0, d1, d2, d3);
    r0 = simd_and(r0, simd_cmp_gt(other, zero));

    simd_vector r1 = all_greater4(zero, d0, d1, d2, d3);
    r1 = simd_and(r1, simd_cmp_lt(other, zero));

    return simd_or(r0, r1);
}

//-----------------------------------------------------------------------------
void process_aabb_triangle(struct simdcol_context* context)
{
    struct aabb_triangle_data* data = context->aabb_triangle;

    if (data->num_items == 0)
        return;

    uint32_t num_vec = (data->num_items + simd_vector_width - 1) / simd_vector_width;
    for(uint32_t vec_index=0; vec_index<num_vec; ++vec_index)
    {
        int offset = vec_index * simd_vector_width;

        simd_vector aabb_min_x = simd_load(data->aabb_min_x + offset);
        simd_vector aabb_min_y = simd_load(data->aabb_min_y + offset);
        simd_vector aabb_max_x = simd_load(data->aabb_max_x + offset);
        simd_vector aabb_max_y = simd_load(data->aabb_max_y + offset);
        simd_vector v0_x = simd_load(data->v_x[0] + offset);
        simd_vector v1_x = simd_load(data->v_x[1] + offset);
        simd_vector v2_x = simd_load(data->v_x[2] + offset);
        simd_vector v0_y = simd_load(data->v_y[0] + offset);
        simd_vector v1_y = simd_load(data->v_y[1] + offset);
        simd_vector v2_y = simd_load(data->v_y[2] + offset);

        // SAT : aabb axis
        simd_vector result = all_less3(aabb_min_x, v0_x, v1_x, v2_x);
        result = simd_or(result, all_greater3(aabb_max_x, v0_x, v1_x, v2_x)); 
        result = simd_or(result, all_less3(aabb_min_y, v0_y, v1_y, v2_y));
        result = simd_or(result, all_greater3(aabb_max_y, v0_y, v1_y, v2_y));

        // SAT : triangle axis
        result = simd_or(result, edge_triangle_aabb(v0_x, v0_y, v1_x, v1_y, v2_x, v2_y, aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y));
        result = simd_or(result, edge_triangle_aabb(v1_x, v1_y, v2_x, v2_y, v0_x, v0_y, aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y));
        result = simd_or(result, edge_triangle_aabb(v2_x, v2_y, v0_x, v0_y, v1_x, v1_y, aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y));

        int bitfield = simd_get_mask(result);

        for(int i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
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
        int offset = vec_index * simd_vector_width;

        simd_vector aabb_min_x = simd_load(data->aabb_min_x + offset);
        simd_vector aabb_min_y = simd_load(data->aabb_min_y + offset);
        simd_vector aabb_max_x = simd_load(data->aabb_max_x + offset);
        simd_vector aabb_max_y = simd_load(data->aabb_max_y + offset);
        simd_vector p0_x = simd_load(data->p0_x + offset);
        simd_vector p0_y = simd_load(data->p0_y + offset);
        simd_vector p1_x = simd_load(data->p1_x + offset);
        simd_vector p1_y = simd_load(data->p1_y + offset);
        simd_vector result = slab_test(aabb_min_x, aabb_max_x, p0_x, p1_x);
        result = simd_and(result, slab_test(aabb_min_y, aabb_max_y, p0_y, p1_y));

        int bitfield = simd_get_mask(result);

        for(int i=0; i<simd_vector_width && bitfield != simd_full_mask; ++i)
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

    free(context);
}
