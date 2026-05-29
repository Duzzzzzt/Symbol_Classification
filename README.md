# Symbol_Classification
CNN Model for symbol classification on c language
## Features:
- Convolution layer
- ReLU
- Softmax
- Cross entropy
- SGD
- SGD + Momentum
- Image loading
- Learning on notMNIST dataset
## Architecture
Input 28x28
↓
Conv 5x5 (8 filters)
↓
ReLU
↓
Flatten
↓
Fully Connected
↓
Softmax

How to run:
1. install notMNIST_small dataset. place it in src/dataset
2. install stb_image.h (https://github.com/nothings/stb/blob/master/stb_image.h) library place it in src/dataset
3. set up preferences on config.txt
4. run gcc -Isrc main.c src/dataset.c src/CNN.c -o main -lm
5. run ./main
6. wait for model to train 
7. for plots run plot.py in plots folder
8. for heatmaps visualisation run heatmap.py in plots folder