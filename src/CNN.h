#ifndef CNN_H
#define CNN_H

typedef struct Tensor{
    float *data;

}Tensor;

void im2col(float *image, int image_width, int image_height, int step,  int conv_size, float *col);
void mat_mult(float *a, float *b, int height_a, int width_a, int width_b, float *out);
int relu(float x);
float **layer(float *image, int kernel_size, int step, int num_of_filters);
float *maxpooling(float *map, int h, int w);
#endif