#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define W 28
#define H 28

typedef struct image {
    unsigned char *data;
    int width, height, channels;
} Image;

typedef struct Images {
    char **paths;
    int cnt;
} Images;

//LOAD IMAGE
Image *load_one_image(const char *image_name) {
    int x, y, n;
    unsigned char *data = stbi_load(image_name, &x, &y, &n, 0);
    if (data == NULL) {
        printf("Cannot open image: %s\n", image_name);
        return NULL;
    }
    
    Image *image = malloc(sizeof(Image));
    image->data = data;
    image->width = x;
    image->height = y;
    image->channels = n;
    

    return image;
}

void free_image(Image *img) {
    if (img) {
        stbi_image_free(img->data);
        free(img);
    }
}

//GET ALL NAMES OF FILES IN DIRECTORY
Images *get_imgs_data(DIR *dir) {
    struct dirent *file;
    
    readdir(dir);
    readdir(dir);
    
    Images *file_names = malloc(sizeof(Images));
    // ONLY ONE FILE CHANGE LATER (REALLOC OR SMTH)
    file_names->paths = malloc(sizeof(char *));
    file_names->cnt = 0;
    
    while ((file = readdir(dir)) != NULL) {

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

int main() {
    struct dirent *files;
    DIR *dir;
    
    dir = opendir("../images");
    if (dir == NULL) {
        printf("Cannot open directory\n");
        return 1;
    }
    
    Images *images = get_imgs_data(dir);
    closedir(dir);
    

    for (int i = 0; i < images->cnt; i++) {
        printf("  %s\n", images->paths[i]);
    }

    Image *image = load_one_image("../images/a.png");

    if (image != NULL) {
        // ПОКА ПЕРВЫЕ 10 ПИКСЕЛЕЙ
        for (int i = 0; i < 10; i++) {
            printf("%d ", image->data[i]);
        }
        printf("\n");
        
        free_image(image);
    }
    
    free_images(images);
    
    return 0;
}