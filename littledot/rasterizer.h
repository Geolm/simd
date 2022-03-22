#ifndef _RASTERIZER_H_
#define _RASTERIZER_H_

struct lines_array;

void lines_array_add(struct lines_array* lines, float x0, float y0, float x1, float y1, float width);


struct rasterizer_context;

struct rasterizer_context* rasterizer_init(int width, int height);
void rasterizer_terminate(struct rasterizer_context* rasterizer);


#endif // _RASTERIZER_H_