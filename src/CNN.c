#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define W 28
#define H 28



void im2col(float *image, int image_width, int image_height, int step, int conv_size, float *col){
    int num_of_pix = conv_size * conv_size;
    int conv_pos_cols = (image_width - conv_size) / step + 1;
    int conv_pos_rows = (image_height - conv_size) / step + 1;
    int final_rows = conv_pos_cols * conv_pos_rows;
    
    for (int row = 0; row < conv_pos_rows; row++) {
        for (int col_idx = 0; col_idx < conv_pos_cols; col_idx++) {
            // OUTPUT MATRIX INDEX
            int out_row = row * conv_pos_cols + col_idx;
            
            
            for (int ky = 0; ky < conv_size; ky++) {
                for (int kx = 0; kx < conv_size; kx++) {
                    // PIXEL COORDS
                    int in_y = row * step + ky;
                    int in_x = col_idx * step + kx;
                    
                    // OUTPUT INDEX
                    int out_col = (ky * conv_size + kx);
                    
                    // INPUT IMAGE INDEX 
                    int image_idx = in_y * image_width + in_x;
                    
                    
                    col[out_row * num_of_pix + out_col] = image[image_idx];
                }
            }
        }
    }
}

void mat_mult(float *a, float *b, int height_a, int width_a, int width_b, float *out) {
    // a: height_a × width_a
    // b: width_a × width_b
    // out: height_a × width_b
    
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
            // FIRST ELEMENT
            float max = map[i * coef * w + j * coef];
            
            // WINDOW COEF x COEF
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

float **layer(float *image, int kernel_size, int step, int num_of_filters){

    srand(time(NULL));  
    float *col;
    int out_w = (W - kernel_size) / step + 1;
    int out_h = (H - kernel_size) / step + 1;
    int final_pix = out_w * out_h;


    col = malloc(final_pix * kernel_size * kernel_size * sizeof(float));

    im2col(image, W, H, step, kernel_size, col);


    float **filter = malloc(num_of_filters * sizeof(float *));


    for(int i = 0; i < num_of_filters; i++){
        filter[i] = malloc(kernel_size * kernel_size * sizeof(float));
        for(int j = 0; j < kernel_size * kernel_size; j ++){
            filter[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }


    float *bias_conv1 = malloc(num_of_filters * sizeof(float));
    for (int f = 0; f < num_of_filters; f++) {
        bias_conv1[f] = 0;
    }


    float **maps = malloc(num_of_filters * sizeof(float *));
    for (int i = 0; i < num_of_filters; i++){
        float *feature_map = malloc(final_pix * sizeof(float));
        // MATRIX MULT FILTERSxCOL
        mat_mult(col,filter[i],final_pix, kernel_size*kernel_size,1, feature_map);
        for (int j = 0; j < final_pix; j++){
            feature_map[j] += bias_conv1[i];
        }
        relu(feature_map, final_pix);

        float *pooled_feature_map = maxpooling(feature_map, out_h, out_w, 2); 
        maps[i] = pooled_feature_map;
        free(feature_map);
    }
    free(bias_conv1);
    free(col);
    for(int i = 0; i < num_of_filters; i++){
        free(filter[i]);
    }
    free(filter);

    return maps;
}