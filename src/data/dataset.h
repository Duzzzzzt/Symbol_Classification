#include <dirent.h>
#ifndef DATASET_H
#define DATASET_H


typedef struct Dataset{
    float *images;
    int* classes;
    int num_of_images;
    int image_size;
}Dataset;

typedef struct image {
    float *data;
    int width, height, channels;
} Image;

typedef struct Images {
    char **paths;
    int cnt;
} Images;


Image *load_one_image(const char *image_name);
void free_image(Image *img);
Images *get_imgs_data(DIR *dir);
void free_images(Images *imgs);
Dataset *load_dataset(const char* path, int max_images);



#endif