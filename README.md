# Song Identifier

A project that identifies songs and artists from lyric snippets using two distinct methods, tested on the Spotify dataset.

## 1. Introduction

This project identifies songs and artists from lyric snippets using two distinct methods, tested on the Spotify dataset (57,650 songs) from [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs/). It includes a Jupyter notebook (`spotify_song_pred.ipynb`) exploring both approaches, separated into standalone Python scripts (`normal_nlp.py` and `bm25_approach.py`) for ease of use.

Example queries like "sing us a song you're the piano man..." and "She's just my kind of girl..." are matched accurately, with BM25 achieving perfect confidence (1.00).

## 2. Approaches

### Normal NLP (Hybrid TF-IDF + Cosine + Jaccard)
- **Description**: A hybrid approach combining TF-IDF vectorization with both Cosine and Jaccard similarity measures to optimize lyric matching.
- **Performance**: Correctly identifies songs (e.g., "Piano Man by Billy Joel" at 0.33 confidence), suitable for exact-match tasks.
- **File**: `normal_nlp.py`
- **Technical Details**:
  1. **TF-IDF Vectorization**:
     * `vectorizer = TfidfVectorizer(ngram_range=(1,3))`: Converts lyrics to TF-IDF vectors, capturing unigrams, bigrams, and trigrams (e.g., "piano," "piano man," "sing us song").
     * `input_vector = vectorizer.transform([input_lyric])`: Turns the query into a TF-IDF vector.
  2. **Cosine Similarity**:
     * `cosine_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()`: Measures similarity between the query's TF-IDF vector and all song vectors, based on angle (cosine) between them.
     * Range: 0 to 1 (1 = identical).
  3. **Jaccard Similarity**:
     * `jaccard_scores = np.array([jaccard_similarity(input_lyric, song_lyric) for song_lyric in cleaned_lyrics])`: Calculates word overlap between the query and each song's cleaned text as a set intersection over union.
     * Range: 0 to 1 (1 = all words match).
  4. **Combined Scores**:
     * `combined_scores = 0.7 * cosine_scores + 0.3 * jaccard_scores`: Blends Cosine (70%) and Jaccard (30%) into a single score.
     * This combination leverages TF-IDF's weighted term importance through Cosine similarity, while Jaccard ensures exact word matches boost the score.

### BM25 (via rank_bm25)
- **Description**: Uses the BM25Okapi algorithm from the rank_bm25 library, tuned with k1=1.5 and b=0.75, optimizing for retrieval by balancing term frequency and lyric length.
- **Performance**: Achieves perfect matches (e.g., "Piano Man by Billy Joel" and "She's My Kind Of Girl by ABBA" at 1.00 confidence), outperforming Normal NLP.
- **File**: `bm25_approach.py`

## 3. Setup

### Cloning the Repository
```bash
git clone https://github.com/ArNAB-0053/Song-Identifier.git
cd Song-Identifier
```

### Installing Dependencies
Manually install the required Python libraries, as no requirements.txt is provided (version compatibility may vary—use latest versions unless issues arise):

* For both approaches:
```bash
pip install pandas numpy nltk
```

* For Normal NLP:
```bash
pip install scikit-learn
```

* For BM25:
```bash
pip install rank_bm25
```

### Dataset
* You can use the `spotify_songs_dataset.csv` file included in this repository.
* If you want the latest version (or if any updates occur), you can download it from [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs/).
* Place the file in the project folder (alongside the scripts) before running the code.

## 4. Running the Code

### Explore Both Approaches
* Open `spotify_song_pred.ipynb` in Jupyter Notebook to see the combined implementation and experimentation:
```bash
jupyter notebook spotify_song_pred.ipynb
```

### Run Individual Scripts
* Normal NLP:
```bash
python normal_nlp.py
```

* BM25:
```bash
python bm25_approach.py
```

* Edit `query`, `query1`, or `query2` in the scripts for custom snippets.
