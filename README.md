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

## How to run:
- unzip notMNIST_small dataset, place it in src/dataset
- set up preferences on config.txt
- run gcc -Isrc main.c src/dataset.c src/CNN.c -o main -lm
- run ./main
- wait for model to train 
- for plots run plot.py in plots folder
- for heatmaps visualisation run heatmap.py in plots folder