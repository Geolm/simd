# simd_collision

This library uses simd.h to compute intersection between 2d primitives with AVX/Neon instructions under the hood. In order to maximize the size of the simd register, the library batches intersection requests and so the results are deferred.


# API

## In a nutshell

* The user provides a callback when initializing the library
* The user calls intersection functions (aabb vs triangle for example) with an user id
* Once the library has enough data, it computes a batch of intersection using simd instructions
* The library calls the callback in case of intersection and pass the user id in parameter

Note : the user can force the library to compute intersection even if the batch is not full (not optimal but sometimes needed)

## Details

```C
struct simdcol_context* simdcol_init(void* user_context, simdcol_callback callback);
```

Init the library, one should pass a pointer to a user context and a valid callback pointer.
The callback has this signature:

```C
typedef void (*simdcol_callback)(void*, uint32_t);
```

The callback is going to be called in case of an intersection with two parameters:
* a void* pointer to the user context provided at initialization
* an uin32_t provided when requesting an intersection test

Every intersection functions take in parameters :
  * the library context (simdcol_context) that contains internal buffers/data
  * an uint32_t user_data that is going to be passed to the callback in case of collision. One could store both primitives id in this user data to run the appropriate code in case of an intersection
  * primitives data (circle center and radius for example)

```C
void simdcol_aabb_triangle(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 p0, vec2 p1, vec2 p2);
void simdcol_aabb_obb(struct simdcol_context* context, uint32_t user_data, aabb box, segment obb_height, float obb_width);
void simdcol_aabb_circle(struct simdcol_context* context, uint32_t user_data, aabb box, vec2 circle_center, float circle_radius);
void simdcol_triangle_triangle(struct simdcol_context* context, uint32_t user_data, const vec2 a[3], const vec2 b[3]);
```

Intersection tests are based on Separate Axis Theorem (or signed distance when a primitive is a circle)

Both winding order are supported for triangles

```C
void simdcol_flush(struct simdcol_context* context, enum flush_hint hint);
```

Force the library to compute intersection even if the batch is not full. The user has to pass a hint on what to flush has described here:

```C
enum flush_hint
{
    flush_aabb_triangle,
    flush_aabb_obb,
    flush_aabb_circle,
    flush_triangle_triangle,
    flush_segment_aabb,
    flush_all
};
```

