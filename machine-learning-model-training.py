# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

tweets_data_path = "preprocessed_tweets.csv"
df = pd.read_csv(tweets_data_path)
print(df.info())
features = df.tweets.values.astype(str)
lables = df[df.columns[4:]].values
print(features[:5])
print(lables[:5])
X_train, X_test, y_train, y_test = train_test_split(
    features, lables, random_state=42, test_size=0.15, shuffle=True)
print("[INFO] Samples from the dataset")
print(X_train[:5])
print(y_train[:5])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X.shape)
print(X_test.shape)

# Creating the SVM model
model = OneVsRestClassifier(SVC())

# Fitting the model with training data
model.fit(X, y_train)

# Making a prediction on the test set
prediction = model.predict(X_test)

# Evaluating the model
print(f"Test Set Accuracy: {accuracy_score(y_test, prediction) * 100} %\n\n")
print(
    f"Classification Report : \n\n{classification_report(y_test, prediction)}")
