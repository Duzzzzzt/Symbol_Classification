#include "dataset.h"
#include "CNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void evaluate(Dataset *test, FullLayer *fc1, FullLayer *fc2){

    int correct = 0;

    for (int s = 0; s < test->num_of_images; s++){

        float *image = &test->images[s * 784];

        int label = test->classes[s];

        float *h1 = full_layer(fc1, image, 784, 128);

        relu(h1, 128);

        float *logits = full_layer(fc2, h1, 128, 10);

        float probs[10];

        softmax(logits, probs, 10);

        if (argmax(probs, 10) == label){
            correct++;
        }

        free(h1);
        free(logits);
    }

    printf("TEST ACCURACY: %.2f%%\n",100.0f * correct / test->num_of_images);
}

void train_epoch_with_momentum(Dataset *train_data, FullLayer *fc1, FullLayer *fc2, float *v_w1, float *v_b1, float *v_w2, float *v_b2, float lr, int batch_size){
     int correct = 0;
    float total_loss = 0;
    
    for (int s = 0; s < (train_data->num_of_images)/batch_size ; s++) {
        float *grad_weights1 = calloc(784 * 128, sizeof(float));
        float *grad_bias1 = calloc(128, sizeof(float));

        float *grad_weights2 = calloc(128 * 10, sizeof(float));
        float *grad_bias2 = calloc(10, sizeof(float));

        float *grad_h1 = malloc(128 * sizeof(float));
        float *grad_input = malloc(784 * sizeof(float));
        for (int i = 0; i < batch_size; i++){
            float *image = &train_data->images[(s * batch_size + i) * 784];
            int label = train_data->classes[(s * batch_size + i)];

            // FULLY CONNECTED
            float *h1 = full_layer(fc1, image, 784, 128);

            relu(h1, 128);
            float *logits = full_layer(fc2, h1, 128, 10);

            float *probs = malloc(10 * sizeof(float));

            softmax(logits, probs, 10);

            // LOSS
            
            float loss = cross_entropy(probs, label, 10);
            total_loss += loss;

            if (argmax(probs, 10) == label) correct++;

            // BACKWARD
            float grad_out[10];
            softmax_cross_entropy_gradient(probs, label, grad_out);

            
            fc_backwards(h1 , fc2->weights, grad_h1, grad_out, grad_weights2, grad_bias2, 128, 10);

            relu_back(grad_h1, h1, 128);

            fc_backwards(image, fc1->weights, grad_input, grad_h1, grad_weights1, grad_bias1, 784, 128);
            free(probs);
            free(h1);
            free(logits);

        }

        // SGD
        sgd_momentum_update(fc1->weights, grad_weights1, v_w1, lr, 0.9f, 784 * 128, batch_size);
        sgd_momentum_update(fc1->bias, grad_bias1, v_b1, lr, 0.9f, 128, batch_size);

        sgd_momentum_update(fc2->weights, grad_weights2, v_w2, lr, 0.9f, 128 * 10, batch_size);
        sgd_momentum_update(fc2->bias, grad_bias2, v_b2, lr, 0.9f, 10, batch_size);

        //FREE ALL
        free(grad_weights1); 
        free(grad_bias1); 
        free(grad_weights2); 
        free(grad_bias2); 
        free(grad_h1);
        free(grad_input);


    }
    printf("Loss: %.4f Acc: %.2f%%\n",total_loss / train_data->num_of_images,100.0f * correct / train_data->num_of_images);
   
}
void train_epoch(Dataset *train_data, FullLayer *fc1, FullLayer *fc2, float lr, int batch_size) {

    int correct = 0;
    float total_loss = 0;
    
    for (int s = 0; s < (train_data->num_of_images)/batch_size ; s++) {
        float *grad_weights1 = calloc(784 * 128, sizeof(float));
        float *grad_bias1 = calloc(128, sizeof(float));

        float *grad_weights2 = calloc(128 * 10, sizeof(float));
        float *grad_bias2 = calloc(10, sizeof(float));

        float *grad_h1 = malloc(128 * sizeof(float));
        float *grad_input = malloc(784 * sizeof(float));
        for (int i = 0; i < batch_size; i++){
            float *image = &train_data->images[(s * batch_size + i) * 784];
            int label = train_data->classes[(s * batch_size + i)];

            // FULLY CONNECTED
            float *h1 = full_layer(fc1, image, 784, 128);

            relu(h1, 128);
            float *logits = full_layer(fc2, h1, 128, 10);

            float *probs = malloc(10 * sizeof(float));

            softmax(logits, probs, 10);

            // LOSS
            
            float loss = cross_entropy(probs, label, 10);
            total_loss += loss;

            if (argmax(probs, 10) == label) correct++;

            // BACKWARD
            float grad_out[10];
            softmax_cross_entropy_gradient(probs, label, grad_out);

            
            fc_backwards(h1 , fc2->weights, grad_h1, grad_out, grad_weights2, grad_bias2, 128, 10);

            relu_back(grad_h1, h1, 128);

            fc_backwards(image, fc1->weights, grad_input, grad_h1, grad_weights1, grad_bias1, 784, 128);
            free(probs);
            free(h1);
            free(logits);

        }

        // SGD
        sgd_update(fc1->weights, grad_weights1, lr, 784 * 128, batch_size);
        sgd_update(fc1->bias, grad_bias1, lr, 128, batch_size);

        sgd_update(fc2->weights, grad_weights2, lr, 128 * 10, batch_size);
        sgd_update(fc2->bias, grad_bias2, lr, 10, batch_size);

        //FREE ALL
        free(grad_weights1); 
        free(grad_bias1); 
        free(grad_weights2); 
        free(grad_bias2); 
        free(grad_h1);
        free(grad_input);


    }
    printf("Loss: %.4f Acc: %.2f%%\n",total_loss / train_data->num_of_images,100.0f * correct / train_data->num_of_images);
   
}

int main() {
    // MAIN DATASET PATH
    const char *main_path = "src/images/notMNIST_small";
    srand(time(NULL));
    Dataset *ds = load_dataset(main_path, 1000);
    
    Dataset *train = malloc(sizeof(Dataset));
    Dataset *test = malloc(sizeof(Dataset));
    train_test_split(ds, 0.8, train, test);

    FullLayer *fc1 = init_weights_full(784, 128);
    FullLayer *fc2 = init_weights_full(128, 10);
    ConvLayer *conv1 = init_weights_conv(6, 1 * 5 * 5);
    ConvLayer *conv2 = init_weights_conv(16, 6 * 5 * 5);

    float *v_w1 = calloc(784 * 128, sizeof(float));
    float *v_b1 = calloc(128, sizeof(float));

    float *v_w2 = calloc(128 * 10, sizeof(float));
    float *v_b2 = calloc(10, sizeof(float));
    
    int epochs = 10;
    float learning_rate = 0.001;

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d:\n", epoch);
        train_epoch(train, fc1, fc2, learning_rate, 32);
        evaluate(test, fc1, fc2);
    }
    
    free_dataset(train);
    free_dataset(test);
    free_dataset(ds);

    return 0;
}
