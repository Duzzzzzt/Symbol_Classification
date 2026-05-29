#include "dataset.h"
#include "CNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    char dataset_path[256];
    int epochs;
    float learning_rate;
    int batch_size;
    int use_momentum;
    float momentum;
    int max_images;
} Config;

void load_config(const char *filename, Config *cfg){

    FILE *file = fopen(filename, "r");

    if(!file){
        printf("Cannot open config file\n");
        exit(1);
    }

    char line[256];

    while(fgets(line, sizeof(line), file)){

        sscanf(line, "DATASET_PATH=%s", cfg->dataset_path);
        sscanf(line, "EPOCHS=%d", &cfg->epochs);
        sscanf(line, "LEARNING_RATE=%f", &cfg->learning_rate);
        sscanf(line, "BATCH_SIZE=%d", &cfg->batch_size);
        sscanf(line, "USE_MOMENTUM=%d", &cfg->use_momentum);
        sscanf(line, "MOMENTUM=%f", &cfg->momentum);
        sscanf(line, "MAX_IMAGES=%d", &cfg->max_images);
    }

    fclose(file);
}
void save_image(float *image, int h, int w, const char *filename){
    FILE *f = fopen(filename, "w");

    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            fprintf(f, "%f ", image[y * w + x]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

void save_heatmaps(float **maps, int filters, int h, int w){

    FILE *file = fopen("heatmap.txt", "a");

    if(!file){
        return;
    }

    for(int f = 0; f < filters; f++){

        fprintf(file, "FILTER %d\n", f);

        for(int y = 0; y < h; y++){

            for(int x = 0; x < w; x++){

                fprintf(file, "%.4f ", maps[f][y * w + x]);
            }

            fprintf(file, "\n");
        }

        fprintf(file, "\n");
    }

    fclose(file);
}

void evaluate(Dataset *test, ConvLayer *conv1, FullLayer *fc){

    int correct = 0;

    for(int s = 0; s < test->num_of_images; s++){
        
        float *image = &test->images[s * 784];
        int label = test->classes[s];

        ConvResult c1 = layer(image, 1, 28, 28, 5, 1, 8, conv1);

        float *flat = flatten(c1.maps, 8, 24, 24);

        float *logits = full_layer(fc, flat, 4608, 10);

        float probs[10];

        softmax(logits, probs, 10);

        if(argmax(probs, 10) == label){
            correct++;
        }

        free(flat);
        free(logits);

        for(int k = 0; k < 8; k++){
            free(c1.maps[k]);
        }

        free(c1.maps);
        free(c1.col);
    }

    printf("TEST ACCURACY: %.2f%%\n", 100.0f * correct / test->num_of_images);
}
void train_epoch_with_momentum(Dataset *train_data, ConvLayer *conv1, FullLayer *fc, float *v_conv1_w, float *v_conv1_b, float *v_fc_w, float *v_fc_b, float lr, int batch_size, float momentum, int epoch){

    int correct = 0;
    float total_loss = 0;
    
    for (int s = 0; s < (train_data->num_of_images)/batch_size ; s++) {

        float *grad_weights2 = calloc(4608 * 10, sizeof(float));
        float *grad_bias2 = calloc(10, sizeof(float));

        float *grad_conv1_weights = calloc(8 * 5 * 5, sizeof(float));
        float *grad_conv1_bias = calloc(8, sizeof(float));

        float *grad_flat2 = malloc(4608  * sizeof(float));


        for (int i = 0; i < batch_size; i++){
            float *image = &train_data->images[(s * batch_size + i) * 784];
            int label = train_data->classes[(s * batch_size + i)];
            
            if(epoch == 29 && s == 0 && i == 0){
                save_image(image, 28, 28, "plots/input.txt");
            }

            // FULLY CONNECTED
            ConvResult c1 = layer(image, 1, 28, 28, 5, 1, 8, conv1);
            if(epoch == 29 && s == 0 && i == 0){
                save_heatmaps(c1.maps, 8, 24, 24);
            }

            float *flat = flatten(c1.maps, 8, 24, 24);
            


            float *logits = full_layer(fc, flat, 4608, 10);

            float *probs = malloc(10 * sizeof(float));

            softmax(logits, probs, 10);

            // LOSS
            
            float loss = cross_entropy(probs, label, 10);
            total_loss += loss;

            if (argmax(probs, 10) == label) correct++;

            // BACKWARD
            float grad_out[10];
            softmax_cross_entropy_gradient(probs, label, grad_out);

            
            fc_backwards(flat, fc->weights, grad_flat2, grad_out, grad_weights2, grad_bias2, 4608, 10);

            for(int f = 0; f < 8; f++){
                conv_backward_weights(c1.col, &grad_flat2[f * 576], grad_conv1_weights + f * 25, &grad_conv1_bias[f], 576, 25);
            }
            // relu_back(grad_flat2, flat2  , 256);
            
            free(probs);
            free(logits);
            free(flat);

            for(int k = 0; k < 8; k++)
                free(c1.maps[k]);

            free(c1.maps);
            

            free(c1.col);
        }

        // SGD
        for(int f = 0; f < 8; f++){
            sgd_momentum_update(conv1->weights[f], grad_conv1_weights + f * 25, v_conv1_w + f * 25, lr, momentum, 25, batch_size);
        }

        sgd_momentum_update(conv1->bias ,grad_conv1_bias, v_conv1_b, lr, momentum, 8, batch_size);

        sgd_momentum_update(fc->weights,grad_weights2,v_fc_w,lr,momentum,4608 * 10,batch_size);
        sgd_momentum_update(fc->bias,grad_bias2,v_fc_b,lr,momentum,10,batch_size);

        //FREE ALL

        free(grad_weights2); 
        free(grad_bias2); 
        free(grad_conv1_weights);
        free(grad_conv1_bias);
        free(grad_flat2);


    }
    printf("Loss: %.4f Acc: %.2f%%\n",total_loss / train_data->num_of_images,100.0f * correct / train_data->num_of_images);
    
    //LOGGING
    FILE *log = fopen("metrics.txt", "a");

    fprintf(log, "%f %f\n", total_loss / train_data->num_of_images, 100.0f * correct / train_data->num_of_images);

    fclose(log);
}



void train_epoch(Dataset *train_data, ConvLayer *conv1,FullLayer *fc, float lr, int batch_size, int epoch){

    int correct = 0;
    float total_loss = 0;
    
    for (int s = 0; s < (train_data->num_of_images)/batch_size ; s++) {

        float *grad_weights2 = calloc(4608 * 10, sizeof(float));
        float *grad_bias2 = calloc(10, sizeof(float));

        float *grad_conv1_weights = calloc(8 * 5 * 5, sizeof(float));
        float *grad_conv1_bias = calloc(8, sizeof(float));

        float *grad_flat2 = malloc(4608  * sizeof(float));


        for (int i = 0; i < batch_size; i++){
            float *image = &train_data->images[(s * batch_size + i) * 784];
            int label = train_data->classes[(s * batch_size + i)];
            if(epoch == 29 && s == 0 && i == 0){
                save_image(image, 28, 28, "plots/input.txt");
            }

            // FULLY CONNECTED
            ConvResult c1 = layer(image, 1, 28, 28, 5, 1, 8, conv1);
            if(epoch == 29 && s == 0 && i == 0){
                save_heatmaps(c1.maps, 8, 24, 24);
            }
            float *flat = flatten(c1.maps, 8, 24, 24);
            


            float *logits = full_layer(fc, flat, 4608, 10);

            float *probs = malloc(10 * sizeof(float));

            softmax(logits, probs, 10);

            // LOSS
            
            float loss = cross_entropy(probs, label, 10);
            total_loss += loss;

            if (argmax(probs, 10) == label) correct++;

            // BACKWARD
            float grad_out[10];
            softmax_cross_entropy_gradient(probs, label, grad_out);

            
            fc_backwards(flat, fc->weights, grad_flat2, grad_out, grad_weights2, grad_bias2, 4608, 10);

            for(int f = 0; f < 8; f++){
                conv_backward_weights(c1.col, &grad_flat2[f * 576], grad_conv1_weights + f * 25, &grad_conv1_bias[f], 576, 25);
            }
            // relu_back(grad_flat2, flat2  , 256);
            
            free(probs);
            free(logits);
            free(flat);

            for(int k = 0; k < 8; k++)
                free(c1.maps[k]);

            free(c1.maps);


            free(c1.col);
        }

        // SGD
        for(int f = 0; f < 8; f++){
            sgd_update(conv1->weights[f], grad_conv1_weights + f * 25, lr, 25, batch_size);
        }

        sgd_update(conv1->bias, grad_conv1_bias, lr, 8, batch_size);

        sgd_update(fc->weights, grad_weights2, lr, 4608 * 10, batch_size);
        sgd_update(fc->bias, grad_bias2, lr, 10, batch_size);

        //FREE ALL

        free(grad_weights2); 
        free(grad_bias2); 
        free(grad_conv1_weights);
        free(grad_conv1_bias);
        free(grad_flat2);

    }
    printf("Loss: %.4f Acc: %.2f%%\n",total_loss / train_data->num_of_images,100.0f * correct / train_data->num_of_images);
    
    //LOGGING
    FILE *log = fopen("metrics.txt", "a");

    fprintf(log, "%f %f\n", total_loss / train_data->num_of_images, 100.0f * correct / train_data->num_of_images);

    fclose(log);
   
}

int main() {
    
    FILE *heatmap = fopen("heatmap.txt", "w");
    fclose(heatmap);
    FILE *m = fopen("metrics.txt", "w");
    fclose(m);

    srand(time(NULL));
    // MAIN DATASET PATH
    Config cfg;
    load_config("config.txt", &cfg);

    printf("CONFIG:\n");
    printf("Dataset: %s\n", cfg.dataset_path);
    printf("Epochs: %d\n", cfg.epochs);
    printf("Learning rate: %f\n", cfg.learning_rate);
    printf("Batch size: %d\n", cfg.batch_size);
    printf("Momentum: %d\n", cfg.use_momentum);

    const char *main_path = cfg.dataset_path;
    
    Dataset *ds = load_dataset(main_path, cfg.max_images);
    printf("Max images = %d\n",cfg.max_images);


    Dataset *train = malloc(sizeof(Dataset));
    Dataset *test = malloc(sizeof(Dataset));
    train_test_split(ds, 0.8, train, test);


    FullLayer *fc = init_weights_full(4608, 10);
    ConvLayer *conv1 = init_weights_conv(8, 1 * 5 * 5);


    float *v_conv1_w = calloc(8 * 25, sizeof(float));
    float *v_conv1_b = calloc(8, sizeof(float));

    float *v_fc_w = calloc(4608 * 10, sizeof(float));
    float *v_fc_b = calloc(10, sizeof(float));

    int epochs = cfg.epochs;
    float learning_rate = cfg.learning_rate;

    if (cfg.use_momentum){
        printf("Using momentum\n");
    } else {
        printf("Standart\n");
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d:\n", epoch);
        if (cfg.use_momentum){
            
            train_epoch_with_momentum(train, conv1, fc, v_conv1_w, v_conv1_b, v_fc_w, v_fc_b, learning_rate, cfg.batch_size, cfg.momentum, epoch);
        } else {
            train_epoch(train, conv1, fc, learning_rate, cfg.batch_size, epoch);
        }
        
        evaluate(test, conv1, fc);
    }
    
    free_dataset(train);
    free_dataset(test);
    free_dataset(ds);

    return 0;
}
