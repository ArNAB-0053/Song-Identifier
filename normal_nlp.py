from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

spotify_file_path ="./spotify_songs_dataset.csv"
data = pd.read_csv(spotify_file_path)

lyrics = data['text'].fillna("").tolist()
song_titles = data['song'].tolist()
artists = data['artist'].tolist()


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens) # Joining the tokens

cleaned_lyrics = [preprocess_text(lyric) for lyric in lyrics]

def jaccard_similarity(text1, text2):
    words_text1 = set(text1.split())
    words_text2 = set(text2.split())
    intersection = words_text1.intersection(words_text2)
    union = words_text1.union(words_text2)
    return len(intersection) / len(union)

# Convert lyrics into TF-IDF vectors
vectorizer = TfidfVectorizer(ngram_range=(1,3))  # Using unigrams, bigrams, and trigrams
tfidf_matrix = vectorizer.fit_transform(cleaned_lyrics)

def find_matching_song(input_lyric, top_k=1):
    input_lyric = preprocess_text(input_lyric)  # Preprocess input text
    input_vector = vectorizer.transform([input_lyric])  # Convert to TF-IDF vector

    # Compute cosine similarity
    cosine_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # Compute Jaccard similarity
    jaccard_scores = np.array([jaccard_similarity(input_lyric, song_lyric) for song_lyric in cleaned_lyrics])

    # Combine Cosine & Jaccard Scores (Weighted)
    combined_scores = 0.7 * cosine_scores + 0.3 * jaccard_scores

    # Get top-k most similar songs
    top_indices = combined_scores.argsort()[-top_k:][::-1]

    # Print
    for idx in top_indices:
        print(f"Matched: {song_titles[idx]} by {artists[idx]} (Confidence: {combined_scores[idx]:.2f})")

query = "sing us a song you're the piano man sing us a song tonight"
find_matching_song(query)