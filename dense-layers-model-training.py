import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks
from nltk import word_tokenize
import pickle

tweets_data_path = "preprocessed_tweets.csv"
df = pd.read_csv(tweets_data_path)
print(df.info())

features = df.tweets.values.astype(str)
lables = df[df.columns[4:]].values

# Fit the tokenizer
# Either pre-define vocab size
# Or get the max possible vocab from text
nltk.download('punkt')
vocab_sz = 500  # None means all
vectorizer = TfidfVectorizer(
    max_features=vocab_sz, tokenizer=word_tokenize, analyzer='word')
vectorizer.fit(features)

X_train, X_test, y_train, y_test = train_test_split(
    features, lables, random_state=42, test_size=0.1, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, random_state=42, test_size=0.1, shuffle=True)

x_train = vectorizer.transform(X_train).toarray()
x_val = vectorizer.transform(X_val).toarray()
x_test = vectorizer.transform(X_test).toarray()


model = Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(vocab_sz,)))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(18, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callback = callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    callbacks=[callback])

# saving deep learning model
model.save('models/dense_model.h5')

# saving tokenizer
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
