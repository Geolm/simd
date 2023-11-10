#include "../simd_2d_collision.h"

void collision_failure(void* user_context, uint32_t user_data)
{
    bool* failure = (bool*) user_context;
    *failure = true;
}

void check_user_data(void* user_context, uint32_t user_data)
{
    bool* failure = (bool*) user_context;
    *failure = (user_data != 0x12345678);
}

TEST collision_init(void)
{
    struct simdcol_context* context = simdcol_init(NULL, collision_failure);
    ASSERT_NEQ(context, NULL);

    simdcol_terminate(context);
    
    PASS();
}

TEST collision_user_data(void)
{
    bool failure = true;
    struct simdcol_context* context = simdcol_init(&failure, check_user_data);
    
    simdcol_aabb_circle(context, 0x12345678, (aabb) {.min = {-40.f, -40.f}, .max = {10.f, 10.f}}, (circle){.center = (vec2){-30.f, -20.f}, .radius = 500.f});

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    
    PASS();
}

TEST aabb_triangle(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    simdcol_aabb_triangle(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, (vec2) {-2.f, 5.f}, (vec2) {-2.f, -5.f});
    simdcol_aabb_triangle(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, (vec2) {2.f, -5.f}, (vec2) {-5.f, 2.f});

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}


SUITE(collision_2d)
{
    RUN_TEST(collision_init);
    RUN_TEST(collision_user_data);
    RUN_TEST(aabb_triangle);
}
