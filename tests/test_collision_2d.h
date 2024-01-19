#include "../extra/simd_2d_collision.h"

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
    
    simdcol_aabb_disc(context, 0x12345678, (aabb) {.min = {-40.f, -40.f}, .max = {10.f, 10.f}}, (vec2){-30.f, -20.f}, 500.f);

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
    
    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);
    
    simdcol_terminate(context);

    PASS();
}

TEST no_intersection_aabb_disc(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    simdcol_aabb_disc(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, 5.f);
    simdcol_aabb_disc(context, 1, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {0.f, -50.f}, 45.f);
    simdcol_aabb_disc(context, 2, (aabb) {.min = {10.f, 10.f}, .max = {20.f, 20.f}}, (vec2) {-5.f, -5.f}, 10.f);

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_aabb_disc(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    simdcol_aabb_disc(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, 10.f);
    simdcol_aabb_disc(context, 1, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {5.f, 5.f}, 25.f);
    simdcol_aabb_disc(context, 2, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {2.f, 2.f}, 5.f);
    simdcol_aabb_disc(context, 3, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {5.f, 8.f}, 5.f);

    simdcol_flush(context, flush_all);

    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);
    
    simdcol_terminate(context);

    PASS();
}

TEST no_intersection_aabb_circle(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    simdcol_aabb_circle(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, 5.f, 2.f);
    simdcol_aabb_circle(context, 1, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {0.f, -50.f}, 45.f, 3.f);
    simdcol_aabb_circle(context, 2, (aabb) {.min = {10.f, 10.f}, .max = {20.f, 20.f}}, (vec2) {-5.f, -5.f}, 10.f, 1.f);
    simdcol_aabb_circle(context, 3, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {0.f, 0.f}, 100.f, 10.f);

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_aabb_circle(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    simdcol_aabb_circle(context, 0, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {5.f, 5.f}, 5.5f, 2.f);
    simdcol_aabb_circle(context, 1, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {5.f, 5.f}, 6.f, 2.f);
    simdcol_aabb_circle(context, 2, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {-5.f, -5.f}, 10.f, 2.f);
    simdcol_aabb_circle(context, 3, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}}, (vec2) {20.f, 20.f}, 30.f, 25.f);

    simdcol_flush(context, flush_all);

    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);
    
    simdcol_terminate(context);

    PASS();
}

TEST no_intersection_segment_aabb(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    simdcol_segment_aabb(context, 0, (vec2) {0.f, 0.f}, (vec2) {1.f, 1.f}, (aabb) {.min = {2.f, 2.f}, .max = {10.f, 10.f}});
    simdcol_segment_aabb(context, 1, (vec2) {0.f, 0.f}, (vec2) {-1.f, -1.f}, (aabb) {.min = {1.f, 1.f}, .max = {10.f, 10.f}});
    simdcol_segment_aabb(context, 2, (vec2) {0.f, 0.f}, (vec2) {1.f, 0.f}, (aabb) {.min = {2.f, -1.f}, .max = {10.f, 10.f}});
    simdcol_segment_aabb(context, 3, (vec2) {0.f, 0.f}, (vec2) {0.f, -10.f}, (aabb) {.min = {2.f, 2.f}, .max = {10.f, 10.f}});
    simdcol_segment_aabb(context, 4, (vec2) {0.f, 0.f}, (vec2) {1.f, 2.f}, (aabb) {.min = {1.f, 0.f}, .max = {2.f, 1.f}});
    simdcol_segment_aabb(context, 5, (vec2) {0.f, 0.f}, (vec2) {1.f, 2.f}, (aabb) {.min = {-1.f, 1.f}, .max = {0.f, 2.f}});
    simdcol_segment_aabb(context, 6, (vec2) {0.f, 0.f}, (vec2) {1.f, 2.f}, (aabb) {.min = {2.f, 4.f}, .max = {3.f, 5.f}});
    simdcol_segment_aabb(context, 7, (vec2) {0.f, 0.f}, (vec2) {1.f, 2.f}, (aabb) {.min = {-2.f, -2.f}, .max = {-1.f, -1.f}});


    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_segment_aabb(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    simdcol_segment_aabb(context, 0, (vec2) {0.f, 0.f}, (vec2) {1.f, 1.f}, (aabb) {.min = {0.f, 0.f}, .max = {10.f, 10.f}});
    simdcol_segment_aabb(context, 1, (vec2) {0.f, 0.f}, (vec2) {-10.f, -10.f}, (aabb) {.min = {-5.f, -5.f}, .max = {-10.f, -10.f}});
    simdcol_segment_aabb(context, 2, (vec2) {0.f, 0.f}, (vec2) {1.f, 0.f}, (aabb) {.min = {0.f, -10.f}, .max = {0.5f, 10.f}});
    simdcol_segment_aabb(context, 3, (vec2) {0.f, 0.f}, (vec2) {0.f, -10.f}, (aabb) {.min = {-500.f, -5.f}, .max = {500.f, -2.f}});

    simdcol_flush(context, flush_all);

    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);
    
    simdcol_terminate(context);

    PASS();
}

TEST no_intersection_segment_disc(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    simdcol_segment_disc(context, 0, (vec2) {0.f, 0.f}, (vec2) {1.f, 1.f}, (vec2) {10.f, 10.f}, 5.f);
    simdcol_segment_disc(context, 1, (vec2) {0.f, 0.f}, (vec2) {100.f, 0.f}, (vec2) {0.f, 20.f}, 5.f);
    simdcol_segment_disc(context, 2, (vec2) {10.f, 0.f}, (vec2) {-100.f, 0.f}, (vec2) {0.f, 20.f}, 5.f);
    simdcol_segment_disc(context, 3, (vec2) {-1.f, -1.f}, (vec2) {-100.f, -200.f}, (vec2) {0.f, 20.f}, 5.f);

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_segment_disc(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    simdcol_segment_disc(context, 0, (vec2) {0.f, 0.f}, (vec2) {1.f, 1.f}, (vec2) {5.f, 5.f}, 10.f);
    simdcol_segment_disc(context, 1, (vec2) {0.f, 0.f}, (vec2) {100.f, 0.f}, (vec2) {0.f, 20.f}, 25.f);
    simdcol_segment_disc(context, 2, (vec2) {10.f, 0.f}, (vec2) {-100.f, 0.f}, (vec2) {-50.f, 20.f}, 25.f);
    simdcol_segment_disc(context, 3, (vec2) {-1.f, -1.f}, (vec2) {-100.f, -100.f}, (vec2) {-50.f, -50.f}, 2.f);

    simdcol_flush(context, flush_all);

    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);

    simdcol_terminate(context);
    PASS();
}

TEST no_intersection_aabb_arc(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    simdcol_aabb_arc(context, 0, (aabb) {.min = {-2.f, 1.f}, .max = {-1.f, 2.f}}, (vec2) {0.f, 0.f}, 1.f, 0.1f, VEC2_PI_2, VEC2_PI_2);
    simdcol_aabb_arc(context, 1, (aabb) {.min = {1.f, 1.f}, .max = {2.f, 2.f}}, (vec2) {0.f, 0.f}, 1.f, 0.1f, VEC2_PI_2, VEC2_PI_2);
    simdcol_aabb_arc(context, 2, (aabb) {.min = {-0.5f, -1.5f}, .max = {1.5f, -0.5f}}, (vec2) {0.f, 0.f}, 1.f, 0.1f, VEC2_PI_2, VEC2_PI_2);
    simdcol_aabb_arc(context, 3, (aabb) {.min = {-1.f, 0.f}, .max = {1.f, 2.f}}, (vec2) {0.f, 0.f}, 4.f, 0.1f, VEC2_PI_2, VEC2_PI_2);
    simdcol_aabb_arc(context, 4, (aabb) {.min = {-1.f, -1.f}, .max = {1.f, 1.f}}, (vec2) {0.f, 0.f}, 4.f, 0.1f, 0.f, VEC2_PI_2);
    simdcol_aabb_arc(context, 5, (aabb) {.min = {-1.f, -1.f}, .max = {1.f, 1.f}}, (vec2) {0.f, 0.f}, 4.f, 0.1f, VEC2_PI, VEC2_PI_2);
    simdcol_aabb_arc(context, 6, (aabb) {.min = {-1.f, -1.f}, .max = {1.f, 1.f}}, (vec2) {0.f, 0.f}, 4.f, 0.1f, VEC2_PI_2, VEC2_PI * 0.75f);
    simdcol_aabb_arc(context, 7, (aabb) {.min = {5.f, 0.f}, .max = {6.f, 1.f}}, (vec2) {0.f, 0.f}, 4.f, 0.1f, VEC2_PI_2, VEC2_PI_2);

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_aabb_arc(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    simdcol_aabb_arc(context, 0, (aabb) {.min = {1.f, 1.f}, .max = {2.f, 2.f}}, (vec2) {0.f, 0.f}, 2.f, 0.1f, VEC2_PI_2, VEC2_PI_4);
    simdcol_aabb_arc(context, 1, (aabb) {.min = {-5.f, -5.f}, .max = {5.f, 5.f}}, (vec2) {0.f, 0.f}, 2.f, 0.1f, VEC2_PI_2, VEC2_PI_2);
    simdcol_aabb_arc(context, 2, (aabb) {.min = {1.f, -1.f}, .max = {2.f, 1.f}}, (vec2) {0.f, 0.f}, 2.f, 0.1f, VEC2_PI_2, VEC2_PI_2);
    simdcol_aabb_arc(context, 3, (aabb) {.min = {-10.f, 1.0f}, .max = {10.f, 3.0f}}, (vec2) {0.f, 0.f}, 2.f, 0.1f, VEC2_PI_2, VEC2_PI_4);

    simdcol_flush(context, flush_all);

    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);

    simdcol_terminate(context);
    PASS();
}

TEST no_intersection_triangle_disc(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    vec2 v0 = {0.f, 4.f}; vec2 v1 = {1.f, 1.f}; vec2 v2 = {5.f, 0.f};

    simdcol_triangle_disc(context, 0, v0, v1, v2, (vec2) {3.f, 3.f}, 1.f);
    simdcol_triangle_disc(context, 1, v0, v1, v2, (vec2) {0.f, 0.f}, 0.9f);
    simdcol_triangle_disc(context, 2, v0, v1, v2, (vec2) {-10.f, -10.f}, 5.f);
    simdcol_triangle_disc(context, 3, v0, v1, v2, (vec2) {4.f, 4.f}, 2.f);

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_triangle_disc(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    vec2 v0 = {0.f, 4.f}; vec2 v1 = {1.f, 1.f}; vec2 v2 = {5.f, 0.f};

    simdcol_triangle_disc(context, 0, v0, v1, v2, (vec2) {2.f, 2.f}, 10.f);
    simdcol_triangle_disc(context, 1, v0, v1, v2, (vec2) {0.f, 0.f}, 2.0f);
    simdcol_triangle_disc(context, 2, v0, v1, v2, (vec2) {7.f, 0.f}, 3.0f);
    simdcol_triangle_disc(context, 3, v0, v1, v2, (vec2) {2.f, 1.f}, 0.1f);

    simdcol_flush(context, flush_all);

    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);

    simdcol_terminate(context);
    PASS();
}

TEST no_intersection_point_triangle(void)
{
    bool failure = false;
    struct simdcol_context* context = simdcol_init(&failure, collision_failure);

    vec2 v0 = {-1.f, 0.f}; vec2 v1 = {0.f, 3.f}; vec2 v2 = {5.f, 0.f};

    simdcol_point_triangle(context, 0, (vec2) {5.f, 2.f}, v0, v1, v2);
    simdcol_point_triangle(context, 1, (vec2) {6.f, 0.f}, v0, v1, v2);
    simdcol_point_triangle(context, 2, (vec2) {-1.f, 3.f}, v0, v1, v2);
    simdcol_point_triangle(context, 3, (vec2) {0.f, -1.f}, v0, v1, v2);

    simdcol_flush(context, flush_all);
    ASSERT_NEQ(failure, true);

    simdcol_terminate(context);
    PASS();
}

TEST intersection_point_triangle(void)
{
    bool success[4] = {false, false, false, false};
    struct simdcol_context* context = simdcol_init(&success, collision_success);

    vec2 v0 = {-1.f, 0.f}; vec2 v1 = {0.f, 3.f}; vec2 v2 = {5.f, 0.f};

    simdcol_point_triangle(context, 0, (vec2) {0.f, 0.1f}, v0, v1, v2);
    simdcol_point_triangle(context, 1, (vec2) {-0.1f, 2.6f}, v0, v1, v2);
    simdcol_point_triangle(context, 2, (vec2) {1.f, 1.f}, v0, v1, v2);
    simdcol_point_triangle(context, 3, (vec2) {-0.5f, 1.f}, v0, v1, v2);

    simdcol_flush(context, flush_all);

    ASSERT(success[0]);
    ASSERT(success[1]);
    ASSERT(success[2]);
    ASSERT(success[3]);

    simdcol_terminate(context);
    PASS();
}


SUITE(collision_2d)
{
    RUN_TEST(collision_init);
    RUN_TEST(collision_user_data);
    RUN_TEST(no_intersection_aabb_triangle);
    RUN_TEST(intersection_aabb_triangle);
    RUN_TEST(no_intersection_aabb_disc);
    RUN_TEST(intersection_aabb_disc);
    RUN_TEST(no_intersection_aabb_circle);
    RUN_TEST(intersection_aabb_circle);
    RUN_TEST(no_intersection_segment_aabb);
    RUN_TEST(intersection_segment_aabb);
    RUN_TEST(no_intersection_segment_disc);
    RUN_TEST(intersection_segment_disc);
    RUN_TEST(no_intersection_aabb_arc);
    RUN_TEST(intersection_aabb_arc);
    RUN_TEST(no_intersection_triangle_disc);
    RUN_TEST(intersection_triangle_disc);
    RUN_TEST(no_intersection_point_triangle);
    RUN_TEST(intersection_point_triangle);
}
