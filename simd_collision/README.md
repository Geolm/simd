# simd_collision

This library uses simd.h to compute intersection between 2d primitives with AVX/Neon instructions under the hood. In order to maximize the size of the simd register, the library batches intersection requests and so the results are deferred.


# API

## In a nutshell

* The user provides a callback when initializing the library
* The user calls intersection functions (aabb vs triangle for example) with an user id
* Once the library has enough intersections, it computes a batch of intersection using simd instructions
* The library calls the callback in case of intersection and pass the used id in parameter

Note : the user can force the library to compute intersection even if the batch is not full (not optimal but sometimes needed)

## Details

struct simdcol_context* simdcol_init(void* user_context, simdcol_callback callback);

Init the library, one should pass a pointer to a user context and a valid callback pointer.
The callback has this signature:

typedef void (*simdcol_callback)(void*, uint32_t);

The callback is going to be called in case of an intersection

Every intersection functions take in parameters :
  * the library context (simdcol_context) that contains internal buffers/data
  * an uint32_t user_data that is going to be passed to the callback in case of collision. One could store both primitives id in this user data to run the appropriate code in case of an intersection
  * primitives data (circle center and radius for example)




