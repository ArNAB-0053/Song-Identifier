import nltk
import string
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

spotify_file_path ="./spotify_songs_dataset.csv"
data = pd.read_csv(spotify_file_path)

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens) # Joining the tokens

lyrics = data['text'].fillna("").apply(preprocess_text).tolist()
song_titles = data['song'].tolist()
artists = data['artist'].tolist()

tokenized_corpus = [lyric.split() for lyric in lyrics]

bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
print("BM25 initialized, corpus size:", len(tokenized_corpus))

def find_matching_song_BM(input_lyric, top_k=1):
    cleaned_lyric = preprocess_text(input_lyric) # Preprocessing
    tokenized_query = cleaned_lyric.split()
    scores = bm25.get_scores(tokenized_query)  # Get BM25 scores
    top_indices = np.argsort(scores)[::-1][:top_k]  # Sort and get top k
    for idx in top_indices:
        confidence = min(scores[idx] / max(scores.max(), 1), 1)  # Normalize to 0-1
        print(f"Matched: {song_titles[idx]} by {artists[idx]} (Confidence: {confidence:.2f})")

query1 = "sing us a song you're the piano man sing us a song tonight"
query2 = "She's just my kind of girl, she makes me feel fine Who could ever believe that"
find_matching_song_BM(query1)
find_matching_song_BM(query2)