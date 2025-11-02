

"""
It takes a DataFrame of historical stock prices and constructs a
NetworkX graph based on the Pearson correlation of their daily returns.
"""

import pandas as pd
import networkx as nx

def build_correlation_graph(prices_df: pd.DataFrame, 
                            corr_method: str = 'pearson',
                            threshold: float = 0.5) -> nx.Graph:
    """
    Builds a weighted, undirected graph from a DataFrame of stock prices.

    The steps are:
    1. Calculate daily percentage returns.
    2. Compute the pairwise correlation matrix (e.g., Pearson).
    3. Create a graph where nodes are stocks.
    4. Add edges between nodes if their *absolute* correlation
       is above the specified threshold.
    5. The edge weight is set to the correlation value.

    Args:
        prices_df: A DataFrame where each column is a stock's historical
                   price (e.g., 'Close') and the index is the Date.
        corr_method: The method to use for correlation ('pearson', 'kendall', 'spearman').
        threshold: The *absolute* correlation value above which an edge
                   will be created. (e.g., 0.5 means |corr| > 0.5).
                   Set to 0.0 to create a fully dense graph.

    Returns:
        A NetworkX graph (nx.Graph) with weighted edges.
    """
    
    print(f"Building correlation graph with threshold={threshold}...")
    
    # 1. Calculate daily percentage returns
    #    pct_change() calculates (current - previous) / previous
    returns_df = prices_df.pct_change().dropna()
    
    # 2. Compute the pairwise correlation matrix
    correlation_matrix = returns_df.corr(method=corr_method)
    
    # 3. Create a graph where nodes are stocks
    G = nx.Graph()
    stocks = correlation_matrix.columns
    G.add_nodes_from(stocks)
    
    # 4. & 5. Add edges based on the correlation and threshold
    #    Iterate through the upper triangle of the matrix to avoid duplicates
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            stock1 = stocks[i]
            stock2 = stocks[j]
            
            # Get the correlation value
            weight = correlation_matrix.loc[stock1, stock2]
            
            # Add the edge *only if* it's above the threshold
            if abs(weight) > threshold:
                G.add_edge(stock1, stock2, weight=weight)
                
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- Demonstration ----------------------------------------------------------

if __name__ == "__main__":
    
    print("--- Starting Correlation Graph Demo ---")
    
    # 1. Create a sample DataFrame of prices
    #    (In a real project, this comes from yfinance)
    sample_data = {
        'Date': pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
            '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'
        ]),
        # AAPL and MSFT are highly correlated
        'AAPL': [150, 152, 151, 153, 155, 154, 156, 158],
        'MSFT': [250, 253, 252, 255, 258, 256, 259, 262],
        # GOOG is correlated with AAPL/MSFT, but less
        'GOOG': [100, 101, 100, 102, 103, 102, 103, 105],
        # F is uncorrelated
        'F':    [12, 11, 12, 11, 12, 11, 12, 11],
        # XOM is negatively correlated
        'XOM':  [110, 109, 110, 108, 107, 108, 106, 105]
    }
    prices_df = pd.DataFrame(sample_data).set_index('Date')
    
    print("\n--- Input Price Data (Head) ---")
    print(prices_df.head())
    
    # --- Demo 1: Sparse Graph with a High Threshold ---
    #    We expect only the strong relationships to appear.
    G_sparse = build_correlation_graph(prices_df, threshold=0.8)
    
    print("\n--- Sparse Graph (Threshold = 0.8) Edges ---")
    for edge in G_sparse.edges(data=True):
        print(f"  {edge}")
        
    # We expect (AAPL, MSFT) and (AAPL, GOOG), (MSFT, GOOG)
    # We expect (F) to be an isolated node (no edges)
    # We expect (XOM) to have strong negative edges
    
    # --- Demo 2: Dense Graph with No Threshold ---
    #    This will connect all nodes, even weak ones.
    G_dense = build_correlation_graph(prices_df, threshold=0.0)
    
    print("\n--- Dense Graph (Threshold = 0.0) Edges (Sample) ---")
    # Print only edges connected to 'AAPL'
    for edge in G_dense.edges('AAPL', data=True):
        print(f"  {edge}")

    print("\n--- Demo Complete ---")
    print("This graph is now ready to be used as an input for the models.")
