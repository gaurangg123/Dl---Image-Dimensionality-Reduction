# Image Dimensionality Reduction and Sentiment Analysis using LSTM with Keras

This project demonstrates the use of LSTM-based autoencoders for image dimensionality reduction and LSTM models for sentiment analysis using Keras. The project consists of two main parts:
1. Dimensionality reduction of the MNIST dataset using an autoencoder.
2. Sentiment analysis on sample text data.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Saving the Models](#saving-the-models)
- [License](#license)

## Introduction

This project showcases the application of autoencoders for reducing the dimensionality of image data and LSTM for sentiment analysis. We use the MNIST dataset for image data and a simple list of sentences for sentiment analysis.

## Installation

1. Clone the repository:
    ```sh
    git clone [https://github.com/yourusername/image-dim-reduction-sentiment-analysis.git](https://github.com/gaurangg123/Dl---Image-Dimensionality-Reduction)
    cd image-dim-reduction-sentiment-analysis
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib tensorflow scikit-learn
    ```

## Usage

1. Run the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
2. Open the `image_dim_reduction_sentiment_analysis.ipynb` file and run all cells.

## Model Architecture

### Autoencoder for Image Dimensionality Reduction

The autoencoder consists of an encoder and a decoder:
- **Encoder**: LSTM layers to encode the input images into a latent space.
- **Decoder**: LSTM layers to decode the latent representation back to the original image dimensions.

### LSTM for Sentiment Analysis

The sentiment analysis model uses:
- **Embedding Layer**: Dense layer to encode the input sequences.
- **LSTM Layer**: LSTM layer to process the sequence data.
- **Output Layer**: Dense layer with a sigmoid activation function for binary classification.

## Training

### Autoencoder Training

- The autoencoder is trained on the MNIST dataset.
- Training parameters: 50 epochs, batch size of 256.

### Sentiment Analysis Model Training

- The sentiment analysis model is trained on sample sentiment data.
- Training parameters: 10 epochs, batch size of 16.

## Results

### Autoencoder Training Loss
![Autoencoder Training Loss](autoencoder_loss.png)

### Sentiment Analysis Accuracy
![Sentiment Analysis Accuracy](sentiment_accuracy.png)

## Saving the Models

Both the autoencoder and sentiment analysis models are saved in the current directory:
- `autoencoder_model.h5`
- `sentiment_model.h5`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
