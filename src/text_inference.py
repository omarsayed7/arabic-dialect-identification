import pickle
import argparse
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import text_cleaning
CLASS_DICT = {0: "AE", 1: "BH", 2: "DZ", 3: "EG", 4: "IQ", 5: "JO", 6: "KW", 7: "LB", 8: "LY", 9: "MA",
              10: "OM", 11: "PL", 12: "QA", 13: "SA", 14: "SD", 15: "SY", 16: "TN", 17: "YE"}


def dense_nn_inference(text, tokenized_path, model_path):
    '''
    Individual testing of the MLP model given a text.
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
    '''
    Selecting the model you want to inference with
    '''
    if model == 'Dense_NN':
        print("[INFO] Inferencing with MLP Network")
        pred_class = dense_nn_inference(
            text, 'models/tokenizer.pickle', "models/dense_model.h5")
    if model == 'LSTM':
        print("[INFO] Inferencing with LSTM Network")
        pred_class = lstm_inference(
            text, 'models/lstm_tokenizer.pickle', "models/lstm_best_model.h5")
    if model == 'SGD':
        print("[INFO] Inferencing with Linear model with SGD learning")
        pred_class = classif_ml_inference(
            text, "models/SGD_model.pkl")
    if model == 'MNb':
        print("[INFO] Inferencing with Multinomial Naive bayes")
        pred_class = classif_ml_inference(
            text, "models/MNb_model.pkl")
    if model == 'RF':
        print("[INFO] Inferencing with Random Forest")
        pred_class = classif_ml_inference(
            text, "models/RF_model.pkl")
    return pred_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Configuration of inference process")
    parser.add_argument('-m', '--model', type=str,
                        help='Choose the model you want to inference with from Dense_NN, MNb, RF, and SGD')
    parser.add_argument('-t', '--text', type=str,
                        help='The text you want to classify')
    args = parser.parse_args()
    print(select_model(args.model, args.text))
