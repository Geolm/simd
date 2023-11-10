#include "../simd_2d_collision.h"

#define UNUSED(x) (void)(x)

void collision_failure(void* user_context, uint32_t user_data)
{
    bool* failure = (bool*) user_context;
    *failure = true;
    UNUSED(user_data);
}

void collision_success(void* user_context, uint32_t user_data)
{
    bool* success = (bool*) user_context;
    success[user_data] = true;
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

TEST no_intersection_aabb_triangle(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    simdcol_aabb_triangle(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, (vec2) {-2.f, 5.f}, (vec2) {-2.f, -5.f});
    simdcol_aabb_triangle(context, 1, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, (vec2) {2.f, -5.f}, (vec2) {-5.f, 2.f});
    simdcol_aabb_triangle(context, 2, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {15.f, -5.f}, (vec2) {10.f, -5.f}, (vec2) {15.f, 2.f});
    simdcol_aabb_triangle(context, 3, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {0.f, 15.f}, (vec2) {10.f, 12.f}, (vec2) {15.f, 15.f});

    
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_aabb_triangle(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    simdcol_aabb_triangle(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {1.f, 1.f}, (vec2) {3.f, 1.f}, (vec2) {1.f, -3.f});
    simdcol_aabb_triangle(context, 1, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-100.f, -100.f}, (vec2) {100.f, 100.f}, (vec2) {-100.f, -100.f});
    simdcol_aabb_triangle(context, 2, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-10.f, -10.f}, (vec2) {2.f, 5.f}, (vec2) {-10.f, 10.f});
    simdcol_aabb_triangle(context, 3, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {5.f, -10.f}, (vec2) {0.f, 20.f}, (vec2) {10.f, 20.f});
    
    simdcol_flush(context, flush_all);
    
    ASSERT_EQ(success[0]&&success[1]&&success[2]&&success[3], true);
    
    simdcol_terminate(context);

    PASS();
}


SUITE(collision_2d)
{
    RUN_TEST(collision_init);
    RUN_TEST(collision_user_data);
    RUN_TEST(no_intersection_aabb_triangle);
    RUN_TEST(intersection_aabb_triangle);
}
