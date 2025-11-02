# textual_data_processing.py

"""
It includes:
1. Text cleaning (lowercase, remove punctuation, numbers, HTML).
2. Tokenization.
3. Stop-word removal.
4. Lemmatization.
5. TF-IDF Vectorization for ML models.
6. Ticker/Entity extraction for building the causation graph.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Setup Function ---------------------------------------------------------

def download_nltk_data():
    """
    Downloads the necessary NLTK datasets for processing.
    You only need to run this once.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading 'punkt' tokenizer data...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpus/stopwords')
    except LookupError:
        print("Downloading 'stopwords' data...")
        nltk.download('stopwords')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading 'wordnet' lemmatizer data...")
        nltk.download('wordnet')
    
    print("NLTK data is ready.")

# --- Main Processing Pipeline -----------------------------------------------

# Initialize components (globally for efficiency)
# We use 'english' stopwords.
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(raw_text: str) -> str:
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes a raw text string.
    This prepares the text for vectorization.

    Args:
        raw_text: A string (e.g., a news article).

    Returns:
        A single string of processed, space-separated tokens.
    """
    
    # 1. Cleaning:
    # Remove HTML tags (if any)
    text = re.sub(r'<[^>]+>', ' ', raw_text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Tokenization:
    tokens = word_tokenize(text)
    
    processed_tokens = []
    # 3. Stop-word removal & 4. Lemmatization:
    for word in tokens:
        if word not in STOP_WORDS and len(word) > 1:
            # Lemmatize the word
            lemma = LEMMATIZER.lemmatize(word)
            processed_tokens.append(lemma)
            
    # 5. Join back into a single string for the vectorizer
    return ' '.join(processed_tokens)

# --- Functions for Project Goals --------------------------------------------

def vectorize_corpus(processed_corpus: list[str], max_features=5000):
    """
    Converts a list of preprocessed text documents into a TF-IDF matrix.
    This matrix is used as features for the ML/DL models.

    Args:
        processed_corpus: A list of processed strings from preprocess_text().
        max_features: The maximum number of top features (words) to keep.

    Returns:
        A tuple of (tfidf_matrix, vectorizer_object)
    """
    print(f"Vectorizing corpus into {max_features} features...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Fit the vectorizer to the corpus and transform the corpus
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)
    
    return tfidf_matrix, vectorizer

def find_mentioned_tickers(raw_text: str, ticker_list: list[str]) -> set[str]:
    """
    Finds all unique stock tickers mentioned in a raw text document.
    This is used to build the edges of the **Causation Graph**.

    Args:
        raw_text: The original, unprocessed news article.
        ticker_list: A list of tickers to search for (e.g., ['AAPL', 'MSFT', 'GOOG']).

    Returns:
        A set of unique tickers found in the text.
    """
    # We use a set for a fast, case-sensitive lookup
    # We also add variations (e.g., $AAPL)
    search_terms = set(ticker_list)
    for ticker in ticker_list:
        search_terms.add(f'${ticker}') # For cashtags
        
    # Use regex to find all matches as whole words
    # \b ensures we match 'AAPL' but not 'SNAPPLE'
    text_content = raw_text.upper() # Standardize case for comparison
    
    found_tickers = set()
    for ticker in ticker_list: # Check against the base list
        # Create a regex to find the ticker as a whole word
        if re.search(r'\b' + re.escape(ticker) + r'\b', text_content):
            found_tickers.add(ticker)
        # Check for cashtag version, e.g., $AAPL
        if re.search(r'\$' + re.escape(ticker) + r'\b', text_content):
            found_tickers.add(ticker)

    return found_tickers

# --- Demonstration ----------------------------------------------------------

if __name__ == "__main__":
    
    print("--- Starting Textual Data Processing Demo ---")
    
    # 1. Ensure NLTK data is available
    download_nltk_data()
    
    # 2. Define our sample data
    sample_corpus = [
        "<html><body><p>Wow! Apple (AAPL) stock is up 5% on news of the new iPhone.</p></body></html>",
        "Analysts are bullish on Microsoft's ($MSFT) new AI projects, but Google (GOOG) is still a leader.",
        "Breaking: $AAPL and $GOOG are rumored to be in a joint venture. MSFT is falling.",
        "This is an irrelevant article about sports."
    ]
    
    # This list would come from your Data Collection module
    ALL_TICKERS = ['AAPL', 'MSFT', 'GOOG']
    
    # --- Demo 1: Preprocessing for ML Features ---
    print("\n--- Demo 1: Preprocessing Text for ML Models ---")
    processed_corpus = []
    for doc in sample_corpus:
        processed_doc = preprocess_text(doc)
        processed_corpus.append(processed_doc)
        print(f"Original: {doc[:60]}...")
        print(f"Processed: {processed_doc}\n")
        
    # --- Demo 2: Vectorizing for ML Features ---
    print("\n--- Demo 2: Vectorizing Corpus (TF-IDF) ---")
    tfidf_matrix, vectorizer = vectorize_corpus(processed_corpus)
    
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} (Documents, Features)")
    print("Feature Names (sample):", vectorizer.get_feature_names_out()[:10])
    
    # --- Demo 3: Finding Tickers for Causation Graph ---
    print("\n--- Demo 3: Finding Tickers for Causation Graph ---")
    
    # This list will store tuples of (ticker, ticker) for graph edges
    edges = []
    
    for doc in sample_corpus:
        tickers_in_doc = find_mentioned_tickers(doc, ALL_TICKERS)
        
        print(f"Document: '{doc[:60]}...'")
        print(f"Found Tickers: {tickers_in_doc}")
        
        # If more than one ticker is mentioned, create edges between them
        if len(tickers_in_doc) > 1:
            from itertools import combinations
            # Create all pairs (e.g., (A,B), (A,C), (B,C))
            for ticker_a, ticker_b in combinations(tickers_in_doc, 2):
                edges.append(tuple(sorted((ticker_a, ticker_b))))
                print(f"  -> Added graph edge: ({ticker_a}, {ticker_b})")
        print("-" * 20)
        
    print("\nAll potential edges for Causation Graph:")
    print(edges)
    # In a real project, you would count these edges to set graph weights
    
    print("\n--- Demo Complete ---")
