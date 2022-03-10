# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk import word_tokenize
import pickle

nltk.download('punkt')

tweets_data_path = "preprocessed_tweets.csv"
df = pd.read_csv(tweets_data_path)
print(df.info())
features = df.tweets.values.astype(str)
lables = df[df.columns[4:]].values
print(features[:5])
print(lables[:5])
X_train, X_test, y_train, y_test = train_test_split(
    features, lables, random_state=42, test_size=0.1, shuffle=True)
print("[INFO] Samples from the dataset")
print(X_train[:5])
print(y_train[:5])

vocab_size = 25000
vectorizer = TfidfVectorizer(
    max_features=vocab_size, tokenizer=word_tokenize, analyzer='word')
X = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X.shape)
print(X_test.shape)

# Creating the Random Forest Classifier model
random_forest = RandomForestClassifier(criterion='gini',
                                       n_estimators=15,
                                       random_state=1,
                                       verbose=2)
# Fitting the model with training data
model = random_forest.fit(X, y_train)

# Making a prediction on the test set
prediction = model.predict(X_test)

# Evaluating the model
print(f"Test Set Accuracy: {accuracy_score(y_test, prediction) * 100} %\n\n")
print(
    f"Classification Report : \n\n{classification_report(y_test, prediction)}")

pickle.dump(model, open("models/Rf_model.sav", 'wb'))
print("Model saved")
