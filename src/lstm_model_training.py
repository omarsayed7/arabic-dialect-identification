import numpy as np
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from sklearn.metrics import classification_report

# Training and tokenization configuration
vocab_size = 15000
max_length = 250
epochs = 1
output_dim = 100
batch_size = 64
#########################################

tweets_data_path = "data/preprocessed_tweets.csv"
df = pd.read_csv(tweets_data_path)
print(df.info())

# Preparing the dataset
features = df.tweets.values.astype(str)
lables = pd.get_dummies(df['dialect']).values  # One-hot encoding the lables

tokenizer = Tokenizer(num_words=vocab_size,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=False)
tokenizer.fit_on_texts(features)

X_train, X_test, y_train, y_test = train_test_split(
    features, lables, random_state=42, test_size=0.1, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, random_state=42, test_size=0.1, shuffle=True)

X_train_tok = tokenizer.texts_to_sequences(X_train)
X_train_tok = pad_sequences(X_train_tok, maxlen=max_length)
print('[INFO] Shape of data tensor:', X_train_tok.shape)

X_val_tok = tokenizer.texts_to_sequences(X_val)
X_val_tok = pad_sequences(X_val_tok, maxlen=max_length)
print('[INFO] Shape of data tensor:', X_val_tok.shape)

X_test_tok = tokenizer.texts_to_sequences(X_test)
X_test_tok = pad_sequences(X_test_tok, maxlen=max_length)
print('[INFO] Shape of data tensor:', X_test_tok.shape)

# LSTM Model
model = models.Sequential()
model.add(Embedding(vocab_size, output_dim, input_shape=(max_length,)))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(18, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Save weights every epoch.
checkpoint = ModelCheckpoint("models/lstm_best_model.h5", monitor='loss',
                             verbose=1, save_best_only=True, mode='auto', period=1)

# Start training
history = model.fit(X_train_tok, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val_tok, y_val), callbacks=[checkpoint])

# saving deep learning model
model.save('models/lstm_best_model.h5')
print("[INFO] Model saved")
print("[INFO] Calculating the classification report")
y_pred = model.predict(X_test_tok, batch_size=512, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(
    f"Classification Report : \n\n{classification_report(y_test, y_pred_bool)}")
