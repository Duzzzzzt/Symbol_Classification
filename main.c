#include "dataset.h"
#include "CNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void train_epoch(Dataset *train_data, FullLayer *fc, ConvLayer *conv1, ConvLayer *conv2, float lr) {
    
    int correct = 0;
    float total_loss = 0;

    for (int s = 0; s < train_data->num_of_images; s++) {

        float *image = &train_data->images[s * 784];
        
        // CONV 1
        float **c1_out = layer(image, 1, 28, 28, 5, 1, 6, conv1);
        float *flat1 = flatten(c1_out, 6, 12, 12);

        // CONV 2
        float **c2_out = layer(flat1, 6, 12, 12, 5, 1, 16, conv2);
        float *flat2 = flatten(c2_out, 16, 4, 4);

        // FULLY CONNECTED
        float *logits = full_layer(fc, flat2, 256, 10);
        float *probs = malloc(10 * sizeof(float));
        softmax(logits, probs, 10);
        
        // LOSS
        int label = train_data->classes[s];
        float loss = cross_entropy(probs, label, 10);
        total_loss += loss;

        if (argmax(probs, 10) == label) correct++;

        // BACKWARD
        float grad_out[10];
        softmax_cross_entropy_gradient(probs, label, grad_out);

        float *grad_weights = malloc(256 * 10 * sizeof(float));
        float *grad_bias = malloc(10 * sizeof(float));
        float *grad_input = malloc(256 * sizeof(float));
        
        fc_backward(grad_out, flat2, fc->weights, grad_weights, grad_bias, grad_input, 256, 10);

        // SGD
        sgd_update(fc->weights, grad_weights, 256 * 10, lr, 1.0);
        sgd_update(fc->bias, grad_bias, 10, lr, 1.0);
        
        // FREE ALL
        free(grad_weights); 
        free(grad_bias); 
        free(grad_input);
        free(probs);
        free(logits); 
        free(flat2); 
        free(flat1);
        
        for (int i = 0; i < 6; i++){    
            free(c1_out[i]);
        }
        free(c1_out);

        for (int i = 0; i < 16; i++){
            free(c2_out[i]);
        }
        free(c2_out);
    }
    
    printf("Loss: %.4f, Acc: %.2f%%\n", total_loss / train_data->num_of_images, 100.0 * correct / train_data->num_of_images);
}

int main() {
    // MAIN DATASET PATH
    const char *main_path = "D:/programming/c_projects/symbol class/notMNIST_small";
    
    Dataset *ds = load_dataset(main_path, 100);
    
    Dataset *train = malloc(sizeof(Dataset));
    Dataset *test = malloc(sizeof(Dataset));
    train_test_split(ds, 0.8, train, test);

    ConvLayer *conv1 = init_weights_conv(6, 1 * 5 * 5);
    ConvLayer *conv2 = init_weights_conv(16, 6 * 5 * 5);
    FullLayer *fc = init_weights_full(256, 10);
    
    int epochs = 20;
    float learning_rate = 0.001;

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d:\n", epoch);
        train_epoch(train, fc, conv1, conv2, learning_rate);
    }
    
    free_dataset(train);
    free_dataset(test);
    free_dataset(ds);

    return 0;
}