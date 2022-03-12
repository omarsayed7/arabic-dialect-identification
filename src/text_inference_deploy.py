import pickle
import argparse
import numpy as np
from tensorflow.keras import models
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.data_preprocessing import text_cleaning

CLASS_DICT = {0: "AE", 1: "BH", 2: "DZ", 3: "EG", 4: "IQ", 5: "JO", 6: "KW", 7: "LB", 8: "LY", 9: "MA",
              10: "OM", 11: "PL", 12: "QA", 13: "SA", 14: "SD", 15: "SY", 16: "TN", 17: "YE"}


def dense_nn_inference(text, tokenized_path, model_path):
    '''
    Individual testing of the Desnse Neural Network model given a text.
    '''
    # loading tokenizer
    with open(tokenized_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    # loading dense model
    loaded_model = models.load_model(model_path)
    tokenized_load = tokenizer.transform([text_cleaning(text)]).toarray()
    prediction_load = loaded_model.predict(tokenized_load)
    return CLASS_DICT[np.argmax(prediction_load)]


def lstm_inference(text, tokenized_path, model_path):
    '''
    Individual testing of the LSTM Network model given a text.
    '''
    # loading tokenizer
    with open(tokenized_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    # loading lstm model
    loaded_model = models.load_model(model_path)
    seq = tokenizer.texts_to_sequences([text_cleaning(text)])
    padded = pad_sequences(seq, maxlen=250)
    prediction_load = loaded_model.predict(padded)

    return CLASS_DICT[np.argmax(prediction_load)]


def classif_ml_inference(text, model_path):
    '''
    Individual testing of the classical ML algorithms (Multinomial NB, RF, SGD) model given a text.
    '''
    # loading dense model
    loaded_model = pickle.load(open(model_path, 'rb'))
    prediction_load = loaded_model.predict([text_cleaning(text)])
    return CLASS_DICT[int(prediction_load)]


def select_model(model, text):
    if model == 'Dense_NN':
        pred_class = dense_nn_inference(
            text, 'src/models/tokenizer.pickle', "src/models/dense_model.h5")
    if model == 'LSTM':
        pred_class = lstm_inference(
            text, 'src/models/lstm_tokenizer.pickle', "src/models/lstm_best_model.h5")
    if model == 'SGD':
        pred_class = classif_ml_inference(
            text, "src/models/SGD_model.pkl")
    if model == 'MNb':
        pred_class = classif_ml_inference(
            text, "src/models/MNb_model.pkl")
    if model == 'RF':
        pred_class = classif_ml_inference(
            text, "src/models/RF_model.pkl")
    return pred_class
