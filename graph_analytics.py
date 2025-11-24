

"""

It calculates metrics like Degree, PageRank, and Centrality for each node
(stock) in the graph. These metrics become new features for the ML models.
"""

import networkx as nx
import pandas as pd

def extract_graph_features(G: nx.Graph, prefix: str) -> pd.DataFrame:
    """
    Calculates structural features for every node in the graph.

    Args:
        G: The input NetworkX graph (Correlation or Causation).
        prefix: A string prefix for the column names (e.g., 'CORR_' or 'NEWS_')
                to distinguish between features from the two graphs.

    Returns:
        A DataFrame where the index is the Stock Ticker and columns are
        the calculated graph features.
    """
    print(f"Extracting features from graph ({prefix})...")
    
    # Initialize a dictionary to hold our features
    features = {}

    # 1. Weighted Degree (Strength)
    # The sum of weights of edges connected to the node
    degree = dict(G.degree(weight='weight'))
    features[f'{prefix}degree'] = degree

    # 2. PageRank
    # Measures the importance of a node
    try:
        pagerank = nx.pagerank(G, weight='weight')
        features[f'{prefix}pagerank'] = pagerank
    except Exception as e:
        print(f"Warning: PageRank failed ({e}). using 0s.")
        features[f'{prefix}pagerank'] = {n: 0 for n in G.nodes()}

    # 3. Betweenness Centrality
    # Note: This can be slow on very large graphs.
    # We use weight=None for speed, or distance=1/weight for weighted paths.
    try:
        betweenness = nx.betweenness_centrality(G, weight=None)
        features[f'{prefix}betweenness'] = betweenness
    except Exception as e:
        print(f"Warning: Betweenness failed. using 0s.")
        features[f'{prefix}betweenness'] = {n: 0 for n in G.nodes()}

    # 4. Clustering Coefficient
    # Measures how connected a node's neighbors are to each other
    try:
        clustering = nx.clustering(G, weight='weight')
        features[f'{prefix}clustering'] = clustering
    except Exception as e:
         features[f'{prefix}clustering'] = {n: 0 for n in G.nodes()}

    # Convert the dictionary of dictionaries to a DataFrame
    df_features = pd.DataFrame(features)
    
    # Fill NaN values (for isolated nodes that have no edges) with 0
    df_features = df_features.fillna(0)
    
    print(f"Extracted {len(df_features.columns)} features for {len(df_features)} stocks.")
    return df_features

# --- Demonstration ----------------------------------------------------------

if __name__ == "__main__":
    
    print("--- Starting Graph Analytics Demo ---")
    
    # 1. Create a dummy graph for demonstration
    # In reality, you would import 'build_correlation_graph' or 'build_causation_graph'
    G_demo = nx.Graph()
    
    # Add nodes
    G_demo.add_nodes_from(['AAPL', 'MSFT', 'GOOG', 'F', 'TSLA'])
    
    # Add weighted edges (representing correlation or news co-occurrence)
    # AAPL is central here
    G_demo.add_edge('AAPL', 'MSFT', weight=0.9)
    G_demo.add_edge('AAPL', 'GOOG', weight=0.8)
    G_demo.add_edge('AAPL', 'TSLA', weight=0.6)
    
    # MSFT connected to GOOG
    G_demo.add_edge('MSFT', 'GOOG', weight=0.85)
    
    # F is isolated (no edges) in this example
    
    # 2. Extract features
    # We use the prefix 'CORR_' assuming this is a correlation graph
    df_graph_features = extract_graph_features(G_demo, prefix='CORR_')
    
    print("\n--- Extracted Graph Features ---")
    print(df_graph_features)
    
    # Interpretation:
    # AAPL should have the highest 'CORR_degree' (it has 3 edges).
    # F should have 0 for everything (it is isolated).
    
    print("\n--- How to Merge with your Price Data ---")
    print("You would now merge this DataFrame with your main time-series DataFrame.")
    print("Note: Since these features are static (based on the whole history),")
    print("you simply repeat these values for every day, OR calculate graphs per month (Rolling Window).")
    
    print("\n--- Demo Complete ---")
