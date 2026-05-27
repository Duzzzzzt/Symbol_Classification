#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "CNN.h"
#define W 28
#define H 28

ConvLayer *init_weights_conv(int num_of_filters, int block_size){
    ConvLayer *weights = malloc(sizeof(ConvLayer));

    weights->weights = malloc(num_of_filters * sizeof(float *));
    for(int i = 0; i < num_of_filters; i++){
        weights->weights[i] = malloc(block_size * sizeof(float));
        for(int j = 0; j < block_size; j++){
            weights->weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.0001f;
        }
    }

    weights->bias = malloc(num_of_filters * sizeof(float));
    for (int f = 0; f < num_of_filters; f++) {
        weights->bias[f] = 0;
    }

    return weights;
}

FullLayer *init_weights_full(int in_size, int out_size){
    FullLayer *weights = malloc(sizeof(FullLayer));

    weights->weights = malloc(in_size * out_size * sizeof(float));
    for (int i = 0; i < in_size * out_size; i++) {
        weights->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.0001f;
    }

    weights->bias = malloc(out_size * sizeof(float));
    for (int i = 0; i < out_size; i++) {
        weights->bias[i] = 0;
    }
    return weights;
}

void free_weights_conv(ConvLayer *weights, int size){
    for(int i = 0; i < size; i++){
        free(weights->weights[i]);
    }
    free(weights->weights);
    free(weights->bias);
    free(weights);
}

void free_weights_full(FullLayer *weights, int size){
    free(weights->weights);
    free(weights->bias);
    free(weights);
}

void im2col(float *image, int num_of_ch, int image_width, int image_height, int step, int conv_size, float *col){
    int num_of_pix = conv_size * conv_size;
    int conv_pos_cols = (image_width - conv_size) / step + 1;
    int conv_pos_rows = (image_height - conv_size) / step + 1;
    int final_rows = conv_pos_cols * conv_pos_rows;
    
    for (int ch = 0; ch < num_of_ch; ch++) {
        float *channel_data = image + ch * image_width * image_height;
        
        for (int row = 0; row < conv_pos_rows; row++) {
            for (int col_idx = 0; col_idx < conv_pos_cols; col_idx++) {
                int out_row = ch * final_rows + row * conv_pos_cols + col_idx;
                
                for (int ky = 0; ky < conv_size; ky++) {
                    for (int kx = 0; kx < conv_size; kx++) {
                        int in_y = row * step + ky;
                        int in_x = col_idx * step + kx;
                        int out_col = (ky * conv_size + kx) + ch * num_of_pix;
                        int image_idx = in_y * image_width + in_x;
                        
                        col[out_row * (num_of_ch * num_of_pix) + out_col] = channel_data[image_idx];
                    }
                }
            }
        }
    }
}

void mat_mult(float *a, float *b, int height_a, int width_a, int width_b, float *out) {
    for (int i = 0; i < height_a; i++) {
        for (int j = 0; j < width_b; j++) {
            float sum = 0;
            for (int k = 0; k < width_a; k++) {
                sum += a[i * width_a + k] * b[k * width_b + j];
            }
            out[i * width_b + j] = sum;
        }
    }
}

void relu(float *x, int size){
    for (int i = 0; i < size; i++){
        if (x[i] < 0.0){
            x[i] = 0.0;
        }
    }
}

float *maxpooling(float *map, int h, int w, int coef){
    int new_h = h / coef;
    int new_w = w / coef;
    
    float *res = malloc(new_h * new_w * sizeof(float));
    
    for(int i = 0; i < new_h; i++){
        for(int j = 0; j < new_w; j++){
            float max = map[i * coef * w + j * coef];
            
            for(int ky = 0; ky < coef; ky++){
                for(int kx = 0; kx < coef; kx++){
                    int y = i * coef + ky;
                    int x = j * coef + kx;
                    float val = map[y * w + x];
                    if(val > max){
                        max = val;
                    }
                }
            }
            res[i * new_w + j] = max;
        }
    }
    
    return res;
}

float *flatten(float **maps, int number_of_maps, int h, int w){
    float *array = malloc(number_of_maps * h * w * sizeof(float));
    int map_size = h * w;
    for (int i = 0; i < number_of_maps; i++){
        for (int j = 0; j < map_size; j++){
            array[i * map_size + j] = maps[i][j];
        }
    }
    return array;
}

float **layer(float *input, int in_c, int in_h, int in_w, int kernel_size, int step, int num_of_filters, ConvLayer *weights){
    float *col;
    int out_w = (in_w - kernel_size) / step + 1;
    int out_h = (in_h - kernel_size) / step + 1;
    int final_pix = out_w * out_h;
    int block_size = in_c * kernel_size * kernel_size;

    col = malloc(final_pix * block_size * sizeof(float));
    im2col(input, in_c, in_h, in_w, kernel_size, step, col);

    float **maps = malloc(num_of_filters * sizeof(float *));
    for (int i = 0; i < num_of_filters; i++){
        float *feature_map = malloc(final_pix * sizeof(float));
        mat_mult(weights->weights[i], col, 1, block_size, final_pix, feature_map);

        for (int j = 0; j < final_pix; j++){
            feature_map[j] += weights->bias[i];
        }
        relu(feature_map, final_pix);
        for (int j = 0; j < final_pix; j++) {
            if (feature_map[j] > 10) feature_map[j] = 10;
        }
        float *pooled_feature_map = maxpooling(feature_map, out_h, out_w, 2); 
        maps[i] = pooled_feature_map;
        free(feature_map);
    }

    free(col);
    return maps;
}

float *full_layer(FullLayer *weights, float *maps, int in_size, int classes){
    float *logits = malloc(classes * sizeof(float));
    
    for (int i = 0; i < classes; i++){
        logits[i] = weights->bias[i];
        for (int j = 0; j < in_size; j++){
            logits[i] += weights->weights[j * classes + i] * maps[j];
        }
    }
    
    return logits;
}

void softmax(float *input, float *output, int n){
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

float cross_entropy(float *probs, int true_label, int num_classes){
    float eps = 1e-8;
    float p = probs[true_label];
    return -logf(p + eps);
}

void softmax_cross_entropy_gradient(float *probs, int true_class, float *grad_out){
    for (int i = 0; i < 10; i++) {
        grad_out[i] = probs[i];
    }
    grad_out[true_class] -= 1.0;
}

void fc_backward(float *grad_out, float *input, float *weights, float *grad_weights, float *grad_bias, float *grad_input, int in_size, int out_size){
    for (int i = 0; i < out_size; i++){
        grad_bias[i] = grad_out[i];
    }

    for (int i = 0; i < in_size; i++){
        for (int j = 0; j < out_size; j++){
            grad_weights[i * out_size + j] = input[i] * grad_out[j];
        }
    }

    for (int i = 0; i < in_size; i++) {
        float sum = 0;
        for (int j = 0; j < out_size; j++){
            sum += weights[i * out_size + j] * grad_out[j];
        }
        grad_input[i] = sum;
    }
}

int argmax(float *array, int size){
    int max_idx = 0;
    for (int i = 1; i < size; i++){
        if (array[i] > array[max_idx]){
            max_idx = i;
        }
    }
    return max_idx;
}

void sgd_update(float *weights, float *gradients, int size, float lr, float clip) {
    for (int i = 0; i < size; i++) {
        float grad = gradients[i];
        
        if (isnan(grad) || isinf(grad)) {
            continue;
        }
        
        if (grad > clip) grad = clip;
        if (grad < -clip) grad = -clip;
        
        weights[i] -= lr * grad;
        
        if (isnan(weights[i]) || isinf(weights[i])) {
            weights[i] = 0;
        }
    }
}