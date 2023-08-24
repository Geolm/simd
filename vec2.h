#ifndef __VEC2__
#define __VEC2__

#include <math.h>
#include <stdbool.h>

//-----------------------------------------------------------------------------
 #define VEC2_PI     (3.14159265f)
 #define VEC2_PI_2   (1.57079632f)
 #define VEC2_PI_4   (0.78539816f)
 #define VEC2_TAU    (6.28318530f)

//-----------------------------------------------------------------------------
static inline float float_sign(float f) {if (f>0.f) return 1.f; if (f<0.f) return -1.f; return 0.f;}
static inline float float_clamp(float f, float a, float b) {if (f<a) return a; if (f>b) return b; return f;}
static inline float float_square(float f) {return f*f;}
static inline float float_min(float a, float b) {return (a<b) ? a : b;}
static inline float float_max(float a, float b) {return (a>b) ? a : b;}

//-----------------------------------------------------------------------------
typedef struct {float x, y;} vec2;
typedef struct {vec2 p0, p1;} segment;
typedef struct {vec2 min, max;} aabb;
typedef struct {vec2 center; float radius;} circle;

//-----------------------------------------------------------------------------
static inline vec2 vec2_splat(float value) {return (vec2) {value, value};}
static inline vec2 vec2_zero(void) {return (vec2) {0.f, 0.f};}
static inline vec2 vec2_angle(float angle) {return (vec2) {cosf(angle), sinf(angle)};}
static inline vec2 vec2_add(vec2 a, vec2 b) {return (vec2) {a.x + b.x, a.y + b.y};}
static inline vec2 vec2_sub(vec2 a, vec2 b) {return (vec2) {a.x - b.x, a.y - b.y};}
static inline vec2 vec2_scale(vec2 a, float f) {return (vec2) {a.x * f, a.y * f};}
static inline vec2 vec2_skew(vec2 v) {return (vec2) {-v.y, v.x};}
static inline vec2 vec2_mul(vec2 a, vec2 b) {return (vec2){a.x * b.x, a.y * b.y};}
static inline vec2 vec2_div(vec2 a, vec2 b) {return (vec2){a.x / b.x, a.y / b.y};}
static inline float vec2_dot(vec2 a, vec2 b) {return a.x * b.x + a.y * b.y;}
static inline float vec2_sq_length(vec2 v) {return vec2_dot(v, v);}
static inline float vec2_length(vec2 v) {return sqrtf(vec2_sq_length(v));}
static inline float vec2_sq_distance(vec2 a, vec2 b) {return vec2_sq_length(vec2_sub(b, a));}
static inline float vec2_distance(vec2 a, vec2 b) {return vec2_length(vec2_sub(b, a));}
static inline vec2 vec2_reflect(vec2 v, vec2 normal) {return vec2_sub(v, vec2_scale(normal, vec2_dot(v, normal) * 2.f));}
static inline vec2 vec2_min(vec2 v, vec2 op) {return (vec2) {.x = (v.x < op.x) ? v.x : op.x, .y = (v.y < op.y) ? v.y : op.y};}
static inline vec2 vec2_min3(vec2 a, vec2 b, vec2 c) {return vec2_min(a, vec2_min(b, c));}
static inline vec2 vec2_min4(vec2 a, vec2 b, vec2 c, vec2 d) {return vec2_min(a, vec2_min3(b, c, d));}
static inline vec2 vec2_max(vec2 v, vec2 op) {return (vec2) {.x = (v.x > op.x) ? v.x : op.x, .y = (v.y > op.y) ? v.y : op.y};}
static inline vec2 vec2_max3(vec2 a, vec2 b, vec2 c) {return vec2_max(a, vec2_max(b, c));}
static inline vec2 vec2_max4(vec2 a, vec2 b, vec2 c, vec2 d) {return vec2_max(a, vec2_max3(b, c, d));}
static inline vec2 vec2_clamp(vec2 v, vec2 lower_bound, vec2 higher_bound) {return vec2_min(vec2_max(v, lower_bound), higher_bound);}
static inline vec2 vec2_saturate(vec2 v) {return vec2_clamp(v, vec2_splat(0.f), vec2_splat(1.f));}
static inline bool vec2_equal(vec2 a, vec2 b) {return (a.x == b.x && a.y == b.y);}
static inline bool vec2_similar(vec2 a, vec2 b, float epsilon) {return (fabs(a.x - b.x) < epsilon) && (fabs(a.y - b.y) < epsilon);}
static inline vec2 vec2_neg(vec2 a) {return  (vec2){-a.x, -a.y};}
static inline vec2 vec2_abs(vec2 a) {return (vec2) {.x = fabs(a.x), .y = fabs(a.y)};}
static inline bool vec2_all_less(vec2 a, vec2 b) {return (a.x < b.x && a.y < b.y);}
static inline bool vec2_any_less(vec2 a, vec2 b) {return (a.x < b.x || a.y < b.y);}
static inline bool vec2_all_greater(vec2 a, vec2 b) {return (a.x > b.x && a.y > b.y);}
static inline bool vec2_any_greater(vec2 a, vec2 b) {return (a.x > b.x || a.y > b.y);}
static inline vec2 vec2_lerp(vec2 a, vec2 b, float t) {return vec2_add(vec2_scale(a, 1.f - t), vec2_scale(b, t));}
static inline float vec2_cross(vec2 a, vec2 b) {return a.x * b.y - a.y * b.x;}
static inline vec2 vec2_sign(vec2 a) {return (vec2) {float_sign(a.x), float_sign(a.y)};}
static inline vec2 vec2_pow(vec2 a, vec2 b) {return (vec2) {powf(a.x, b.x), powf(a.y, b.y)};}
static inline float vec2_atan2(vec2 v) {return atan2f(v.y, v.x);}
static inline void vec2_swap(vec2* a, vec2* b) {vec2 tmp = *a; *a = *b; *b = tmp;}
static inline vec2 vec2_rotate(vec2 point, vec2 rotation) {return (vec2) {point.x * rotation.x - point.y * rotation.y, point.y * rotation.x + point.x * rotation.y};}
static inline vec2 vec2_quadratic_bezier(vec2 p0, vec2 p1, vec2 p2, float t) {float omt = 1.f-t;return vec2_add(vec2_add(vec2_scale(p0, omt * omt), vec2_scale(p1, 2.f * omt * t)), vec2_scale(p2, t * t));}
static inline float vec2_normalize(vec2* v)
{
    float norm = vec2_length(*v);
    if (norm == 0.f)
        return 0.f;

    *v = vec2_scale(*v, 1.f / norm);
    return norm;
}

// based on https://en.wikipedia.org/wiki/Alpha_max_plus_beta_min_algorithm
// + one newton iteration
static inline float vec2_approx_length(vec2 a)
{
    vec2 v_abs = vec2_abs(a);
    float min_value, max_value;

    if (v_abs.x > v_abs.y)
    {
        min_value = v_abs.y;
        max_value = v_abs.x;
    }
    else
    {
        min_value = v_abs.x;
        max_value = v_abs.y;
    }

    float approximation = 0.898204193266868f * max_value + 0.485968200201465f * min_value;
    if (max_value > approximation)
        approximation = max_value;

    return (approximation + vec2_sq_length(a)/approximation) / 2.f;
}

#endif 
