import pandas as pd
import re
import string
import pyarabic.araby as araby
import nltk
from nltk.corpus import stopwords
import textblob
from sklearn import preprocessing

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def text_cleaning(text):
    # Removing Punctuations and Symbols
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)
    # Remove Emojis
    text = emoji_pattern.sub(r'', text)
    return text


def text_normalization(text):
    text = re.sub(arabic_diacritics, '', text)
    text = araby.strip_diacritics(text)
    text = araby.strip_shadda(text)
    text = araby.strip_tashkeel(text)
    # Remove non-arabic chars
    text = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = re.sub('([@A-Za-z0-9_ـــــــــــــ]+)|[^\w\s]|#|http\S+', ' ', text)
    text = re.sub(r'\\u[A-Za-z0-9\\]+', ' ', text)
    # Remove repeated letters
    text = text.strip()
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r'(.)\1+', r'\1', text)
    return text


def text_preprocessing(dataFrame, text_column):
    print("[INFO] Starting of pre-processing on the text")
    dataFrame[text_column] = dataFrame[text_column].astype(str)
    print("[INFO] Starting with text cleaning on the text")
    dataFrame[text_column] = dataFrame[text_column].apply(text_cleaning)
    print("[INFO] Finishing with text cleaning on the text")

    # Remove stop words
    nltk.download('stopwords')
    stop = stopwords.words('arabic')
    dataFrame[text_column] = dataFrame[text_column].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop))

    print("[INFO] Starting with text normalization on the text")
    dataFrame[text_column] = dataFrame[text_column].apply(text_normalization)
    print("[INFO] Finishing with text normalization on the text")

    # Lemmatisation
    nltk.download('wordnet')
    dataFrame[text_column] = dataFrame[text_column].apply(
        lambda x: " ".join([textblob.Word(word).lemmatize() for word in x.split()]))

    print("[INFO] Finished pre-processing on the text")
    print("[INFO] Last step is encoding the class lables")

    # Encoding the lables (18 classes)
    encoding = preprocessing.LabelEncoder()

    # using fit transform on the data
    y = encoding.fit_transform(dataFrame.dialect.values)

    # replace dialec column with the encoded
    dataFrame['dialect'] = y

    return dataFrame


def inference_cleaning(text):
    text = text_cleaning(text)
    text = text_normalization(text)
    return text


if __name__ == '__main__':
    tweets_data_path = "tweets_dataset.csv"
    df = pd.read_csv(tweets_data_path, encoding='utf-8')
    print(df.info())
    processed_dataFrame = text_preprocessing(df, 'tweets')
    print(processed_dataFrame.info())
    # Save preprocessed dataframe to CSV
    processed_dataFrame.to_csv('preprocessed_tweets.csv')
