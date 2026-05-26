#include "dataset.h"
#include "CNN.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // MAIN DATASET PATH
    const char *main_path = "D:/programming/c_projects/symbol class/notMNIST_small";
    // ONLY 10 IMAGES IN CLASS NOW
    Dataset *ds = load_dataset(main_path,10);

    int *train = malloc(100 * sizeof(int));
    int *test = malloc(100 * sizeof(int));
    train_test_split(ds,0.8,train,test);

    free(train);
    free(test);
    // DEBUG 1 IMAGE
    Image *image = load_one_image("src/images/a.png");

    ConvLayer *cw1 = init_weights_conv(6, 25); 

    float **maps1 = layer(image->data, 1, 28, 28, 5, 1, 6, cw1);
    float *flat1 = flatten(maps1, 6, 12, 12);
    
    ConvLayer *cw2 = init_weights_conv(6, 25); 

    float **maps2 = layer(flat1, 1, 28, 28, 5, 1, 6, cw2);
    float *flat2 = flatten(maps2, 6, 12, 12);

    FullLayer *fw = init_weights_full(256, 10);
    float *logits = full_layer(fw, flat1, 256, 10);
    printf("Probabilities: ");  
    for(int i = 0; i < 10; i++){
        printf("%f ", logits[i]);
    }

    printf("Channels: %d\n", image->channels);
    printf("Width: %d, Height: %d\n", image->width, image->height);
    free_image(image);



    return 0;
}