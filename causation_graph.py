# causation_graph.py

"""

It takes a corpus of raw text documents and a list of stock tickers,
then builds a NetworkX graph based on the co-occurrence frequency
of those tickers in the corpus.

This module *depends* on the `find_mentioned_tickers` function from
`textual_data_processing.py`.
"""

import networkx as nx
from itertools import combinations
from collections import Counter


try:
    from textual_data_processing import find_mentioned_tickers
except ImportError:
    print("Error: Could not import 'find_mentioned_tickers'.")
    print("Please ensure 'textual_data_processing.py' is in the same directory.")
    # Define a dummy function for demonstration purposes if import fails
    def find_mentioned_tickers(raw_text: str, ticker_list: list[str]) -> set[str]:
        print("Warning: Using dummy find_mentioned_tickers function.")
        found = set()
        for ticker in ticker_list:
            if ticker in raw_text:
                found.add(ticker)
        return found


def build_causation_graph(raw_corpus: list[str], ticker_list: list[str]) -> nx.Graph:
    """
    Builds a weighted, undirected graph based on ticker co-occurrence in a
    list of documents (e.g., news articles).

    The edge weight represents the number of times two tickers
    were mentioned together in the same document.

    Args:
        raw_corpus: A list of raw text strings (the documents).
        ticker_list: The master list of stock tickers to look for.

    Returns:
        A NetworkX graph (nx.Graph) with weighted edges.
    """
    print("Building causation graph from news corpus...")
    
    # We use a Counter to store the weights of the edges
    # The key will be a sorted tuple, e.g., ('AAPL', 'MSFT')
    edge_weights = Counter()
    
    # 1. Iterate through every document in the corpus
    for doc in raw_corpus:
        # 2. Find all tickers mentioned in this single document
        tickers_in_doc = find_mentioned_tickers(doc, ticker_list)
        
        # 3. If 2 or more tickers are mentioned, they are connected
        if len(tickers_in_doc) >= 2:
            # Create all unique pairs of tickers, e.g., (A,B), (A,C), (B,C)
            for ticker_a, ticker_b in combinations(tickers_in_doc, 2):
                # Create a canonical (sorted) key for the pair
                key = tuple(sorted((ticker_a, ticker_b)))
                # Increment the count for this pair
                edge_weights[key] += 1
                
    # 4. Create the graph
    G = nx.Graph()
    
    # 5. Add all tickers as nodes (even if they have no edges)
    G.add_nodes_from(ticker_list)
    
    # 6. Add all the weighted edges from our Counter
    for (stock1, stock2), weight in edge_weights.items():
        G.add_edge(stock1, stock2, weight=weight)
        
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- Demonstration ----------------------------------------------------------

if __name__ == "__main__":
    
    print("--- Starting Causation Graph Demo ---")
    
    # 1. Define our sample data (same as textual_data_processing.py)
    sample_corpus = [
        "<html><body><p>Wow! Apple (AAPL) stock is up 5% on news of the new iPhone.</p></body></html>",
        "Analysts are bullish on Microsoft's ($MSFT) new AI projects, but Google (GOOG) is still a leader.",
        "Breaking: $AAPL and $GOOG are rumored to be in a joint venture. MSFT is falling.",
        "This is an irrelevant article about sports.",
        "Another joint venture between Apple (AAPL) and Google (GOOG) was announced."
    ]
    
    ALL_TICKERS = ['AAPL', 'MSFT', 'GOOG']
    
    # 2. Build the graph
    G_causation = build_causation_graph(sample_corpus, ALL_TICKERS)
    
    # 3. Inspect the resulting graph
    print("\n--- Causation Graph Edges & Weights ---")
    
    # Expected output:
    # ('MSFT', 'GOOG') -> weight = 1 (from article 2)
    # ('AAPL', 'GOOG') -> weight = 2 (from articles 3 and 5)
    # ('AAPL', 'MSFT') -> weight = 1 (from article 3)
    
    for edge in G_causation.edges(data=True):
        print(f"  {edge}")
        
    print("\n--- Demo Complete ---")
    print("This graph now models the relationships from the news data.")
