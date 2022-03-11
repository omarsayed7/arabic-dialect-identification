# Arabic Dialect Identification

## Introduction

Many countries speak Arabic; however, each country has its own dialect, the aim of this task is to build a model that predicts the dialect given the text.

## Usage

#### Installing dependencies

    pip install -r requirements.txt

#### Data fetching

    python data_fetching.py

#### Data pre-processing

    python data_preprocessing.py

#### Machine learning and deep learning model training

##### Machine learning model training

    python machine_learning_model_training.py [-m [model_type]]

    --model_type                 Choose the model you want to train from Multinomial NB, RF, and SGD

##### Simple neural network(Dense layers) model training

    python dense_layers_model_training.py

##### LSTM neural network model training

    python lstm_model_training.py

#### Inference the trained models

    python text_inference.py [-m [model_type] -t [text]]

    --model_type                 Choose the model you want to train from LSTM, Dense_NN, SVM, RF, and SGD
    --text                       The text you want to classify
