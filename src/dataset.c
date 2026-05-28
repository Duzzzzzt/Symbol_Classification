#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "dataset.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define W 28
#define H 28


//LOAD IMAGE
Image *load_one_image(const char *image_name) {
    int x, y, n;
    unsigned char *data = stbi_load(image_name, &x, &y, &n, 1);
    n = 1;
    if (data == NULL) {
        printf("Cannot open image: %s\n", image_name);
        return NULL;
    }
    
    Image *image = malloc(sizeof(Image));
    image->width = x;
    image->height = y;
    image->channels = n;

    int pixel_count = x * y;
    image->data = malloc(pixel_count * n * sizeof(float));
    
    for (int i = 0; i < pixel_count * n; i++) {
        ((float*)image->data)[i] = data[i] / 255.0f;
    }
    
    stbi_image_free(data);

    return image;
}

void free_image(Image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

//GET ALL NAMES OF FILES IN DIRECTORY
Images *get_imgs_data(DIR *dir) {
    struct dirent *file;
    
    readdir(dir);
    readdir(dir);
    
    Images *file_names = malloc(sizeof(Images));
    file_names->paths = malloc(sizeof(char *));
    file_names->cnt = 0;
    
    while ((file = readdir(dir)) != NULL) {
        file_names->paths = realloc(file_names->paths, (file_names->cnt + 1) * sizeof(char *));
        file_names->paths[file_names->cnt++] = strdup(file->d_name);
        
    }
    
    return file_names;
}

void free_images(Images *imgs) {
    if (imgs) {
        for (int i = 0; i < imgs->cnt; i++) {
            free(imgs->paths[i]);
        }
        free(imgs->paths);
        free(imgs);
    }
}

Dataset *load_dataset(const char* path, int max_images){
    const char *classes[] = {"A","B","C","D","E","F","G","H","I","J"};
    const int number_of_classes = 10;
    int total_cnt = 0;


    for (int i = 0; i < number_of_classes; i++){
        char *path_for_class = malloc(512);
        snprintf(path_for_class, 512, "%s/%s", path, classes[i]);

        DIR *dir = opendir(path_for_class);
        Images *images = get_imgs_data(dir);

        int count = images->cnt < max_images ? images->cnt : max_images;
        total_cnt += count;


        free_images(images);
        closedir(dir);
    }
    printf("total images = %d\n", total_cnt);

    Dataset *ds = malloc(sizeof(Dataset));
    ds->images = malloc(total_cnt * W * H * sizeof(float));
    ds->image_size = W * H;
    ds->num_of_images = total_cnt;
    ds->classes = malloc(total_cnt * sizeof(int));
    int index = 0;
    // LOAD EVERY CLASS
    for (int class = 0; class < number_of_classes; class++){
        char *path_for_class = malloc(512);
        snprintf(path_for_class, 512, "%s/%s", path, classes[class]);

        DIR *dir = opendir(path_for_class);
        Images *images = get_imgs_data(dir);

        int to_load = images->cnt < max_images ? images->cnt : max_images;

        // LOAD IMAGES FOR EVERY CLASS
        for (int j = 0; j < to_load; j++){
            char image_path[512];
            snprintf(image_path, sizeof(image_path), "%s/%s", path_for_class, images->paths[j]);

            Image *img = load_one_image(image_path);
            if(img == NULL){
                printf("Cannot load image\n");
                continue;
            }
            for (int pixel = 0; pixel < W * H; pixel++){
                ds->images[index * W * H + pixel] = img->data[pixel];
            }

            ds->classes[index] = class;
            index++;
            free_image(img);
        }
        printf("Loaded %d images in %s class\n", to_load, classes[class]);
 
        free_images(images);
        closedir(dir);
    }
    
    return ds;
}

void train_test_split(Dataset *dataset, float ratio, Dataset *train, Dataset *test){
    int *indices = malloc(dataset->num_of_images * sizeof(int));
    for (int i = 0; i < dataset->num_of_images; i++) {
        indices[i] = i;
    }
    for (int i = dataset->num_of_images - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    int num_of_train = dataset->num_of_images * ratio;
    int num_of_test = dataset->num_of_images - num_of_train;
    int global_index = 0;
    printf("Number of train images: %d\n",num_of_train);
    printf("Number of test images: %d\n",num_of_test);

    train->images = malloc(num_of_train * dataset->image_size * sizeof(float));
    train->classes = malloc(num_of_train * sizeof(int));
    train->num_of_images = num_of_train;
    train->image_size = dataset->image_size;
    

    test->images = malloc(num_of_test * dataset->image_size * sizeof(float));
    test->classes = malloc(num_of_test * sizeof(int));
    test->num_of_images = num_of_test;
    test->image_size = dataset->image_size;
    

    for (int i = 0; i < num_of_train; i++) {
        int idx = indices[i];

        memcpy(&train->images[i * dataset->image_size],
               &dataset->images[idx * dataset->image_size],
               dataset->image_size * sizeof(float));

        train->classes[i] = dataset->classes[idx];
    }
    

    for (int i = 0; i < num_of_test; i++) {
        int idx = indices[num_of_train + i];

        memcpy(&test->images[i * dataset->image_size], &dataset->images[idx * dataset->image_size], dataset->image_size * sizeof(float));

        test->classes[i] = dataset->classes[idx];
    }
    
    free(indices);
}

void free_dataset(Dataset *ds){
    free(ds->classes);
    free(ds->images);
    free(ds);
}