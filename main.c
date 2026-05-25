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
    float *out = malloc(576 * 25 * sizeof(float));
    float **maps = malloc(6 * sizeof(float *));
    maps = layer(image->data, 5, 1, 6);
    for(int i = 0; i < 144; i++){
        printf("%f ",maps[0][i]);
    }
    printf("Channels: %d\n", image->channels);
    printf("Width: %d, Height: %d\n", image->width, image->height);
    free_image(image);

    return 0;
}