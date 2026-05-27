#ifndef CNN_H
#define CNN_H

typedef struct {
    float **weights;
    float *bias;
}ConvLayer;


typedef struct{
    float *weights;
    float *bias;
}FullLayer;

ConvLayer *init_weights_conv(int num_of_filters, int block_size);
FullLayer *init_weights_full(int in_size, int out_size);
void free_weights_conv(ConvLayer *weights, int size);
void free_weights_full(FullLayer *weights, int size);
void im2col(float *image, int num_of_ch, int image_width, int image_height, int step, int conv_size, float *col);
void mat_mult(float *a, float *b, int height_a, int width_a, int width_b, float *out);
void relu(float *x, int size);
float **layer(float *input, int in_c, int in_h, int in_w, int kernel_size, int step, int num_of_filters, ConvLayer *weights);
float *maxpooling(float *map, int h, int w, int coef);
float *flatten(float **maps, int number_of_maps, int h, int w);
float *full_layer(FullLayer *weights, float *maps, int in_size, int classes);
float cross_entropy(float *probs, int true_label, int num_classes);
void fc_backward(float *grad_out, float *input, float *weights, float *grad_weights, float *grad_bias, float *grad_input, int in_size, int out_size);

int argmax(float *array, int size);
void softmax_cross_entropy_gradient(float *probs, int true_class, float *grad_out);
void sgd_update(float *weights, float *gradients, int size, float lr, float clip);

#endif