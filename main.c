#include "dataset.h"
#include <stdio.h>

int main() {
    // MAIN DATASET PATH
    const char *main_path = "D:/programming/c_projects/symbol class/notMNIST_small";
    // ONLY 10 IMAGES IN CLASS NOW
    Dataset *ds = load_dataset(main_path,10);



    // DEBUG 1 IMAGE
    Image *image = load_one_image("src/images/a.png");
    printf("Channels: %d\n", image->channels);
    printf("Width: %d, Height: %d\n", image->width, image->height);
    free_image(image);
    

    
    return 0;
}