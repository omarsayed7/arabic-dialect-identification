# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import pickle
import argparse

nltk.download('punkt')


def model_training(model_type, X_train, y_train):
    '''
    Machine learning model training with the whole vocab
    '''
    if model_type == 'SGD':
        print("[INFO] Starting training with SGD Classifier")
        model = Pipeline([('tfidf', TfidfVectorizer()),
                          ('sgd', SGDClassifier(max_iter=1000, tol=1e-3))])
        model.fit(X_train, y_train)
        return model

    if model_type == 'RF':
        print("[INFO] Starting training with Random Forest Classifier")
        model = Pipeline([('tfidf', TfidfVectorizer()),
                          ('clf', RandomForestClassifier(criterion='gini',
                                                         n_estimators=15,
                                                         min_samples_split=10,
                                                         min_samples_leaf=1,
                                                         max_features='auto',
                                                         oob_score=True,
                                                         random_state=1,
                                                         n_jobs=-1,
                                                         verbose=2)),
                          ])
        model.fit(X_train, y_train)
        return model

    if model_type == 'MNb':
        print(
            "[INFO] Starting training with Naive Bayes classifier for multinomial models")
        model = Pipeline([('tfidf', TfidfVectorizer()),
                          ('svm', MultinomialNB())])
        model.fit(X_train, y_train)
        return model
    else:
        print("[ERROR] You entered wrong model name")
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Configuration of setup and training process")
    parser.add_argument('-m', '--model', type=str,
                        help='Choose the model you want to train from Multinomial NB, RF, and SGD')
    args = parser.parse_args()

    # Reading the tweets
    tweets_data_path = "preprocessed_tweets.csv"
    df = pd.read_csv(tweets_data_path)
    print(df.info())

    # Assigning the features and lables from the dataframe
    features = df.tweets.values.astype(str)
    lables = df.dialect.values.astype('float32')
    print(features[:5])
    print(lables[:5])

    # Splitting the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        features, lables, random_state=42, test_size=0.1, shuffle=True)
    print("[INFO] Samples from the dataset")
    print(X_train[:5])
    print(y_train[:5])
    # Start model training
    model = model_training(args.model, X_train, y_train)
    # Making a prediction on the test set
    prediction = model.predict(X_test)
    # Evaluating the model
    print(
        f"Test Set Accuracy: {accuracy_score(y_test, prediction) * 100} %\n\n")
    print(
        f"Classification Report : \n\n{classification_report(y_test, prediction)}")
    # Saving the model
    pickle.dump(model, open("models/"+str(args.model) + "_model.pkl", 'wb'))
    print("Model saved")
