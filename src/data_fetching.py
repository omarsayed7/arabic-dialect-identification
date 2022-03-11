import requests
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

URL = "https://recruitment.aimtechnologies.co/ai-tasks"
tweets_ids_csv = "data/dialect_dataset.csv"
tweets_ids = pd.read_csv(tweets_ids_csv)


def data_fetch(df, col, chunck_size):
    '''
    INPUT: dataframe, column_of_ids, Chunk_size(Not exceed 1000)
    OUTPUT: Tweets
    Description: Making data fetch function to call the API with the specified Ids
    '''
    df = df.astype(str)
    tweets = np.array([])
    lst = df[col].tolist()
    chunks = [lst[i:i + 1000] for i in range(0, len(lst), chunck_size)]
    for chunck in tqdm(chunks):
        json_chunck = json.dumps(chunck)
        r = requests.post(URL, data=json_chunck)
        tweets_chunck = r.json()
        tweets_chunck_list = list(tweets_chunck.values())
        tweets = np.append(tweets, tweets_chunck_list)
    tweets = tweets.flatten()
    return tweets


returned_tweets = data_fetch(tweets_ids, 'id', 1000)
# make a copy of the original dataframe to append
fetched_tweets = tweets_ids.copy()
# append the tweets column in final_df dataframe
fetched_tweets['tweets'] = returned_tweets
# save the final_df to csv file
fetched_tweets.to_csv("data/tweets_dataset.csv")
