 

"""
This module provides functions for creating prediction targets (labels)
for time-series stock data, as described in Module 2.4.

It includes functions for:
1. Classification Target (e.g., 1 for 'Up', 0 for 'Down').
2. Regression Target (e.g., the actual price 'n' days ahead).

Crucially, these functions handle the 'shifting' of data to prevent
lookahead bias.
"""

import pandas as pd
import numpy as np # Used for NaN

def create_classification_target(df: pd.DataFrame, 
                                 price_col: str = 'Close', 
                                 horizon: int = 1,
                                 neutral_threshold: float = 0.005) -> pd.DataFrame:
    """
    Creates a multi-class classification target (Up, Down, Hold).

    - 'Up' (2): Price increased by more than the threshold.
    - 'Hold' (1): Price change was within the threshold.
    - 'Down' (0): Price decreased by more than the threshold.

    Args:
        df: The input DataFrame (must be sorted by date).
        price_col: The name of the column to base the prediction on (e.g., 'Close').
        horizon: The number of days to look ahead (e.g., 1 = predict next day).
        neutral_threshold: The percentage (as a decimal) to consider "neutral".
                           e.g., 0.005 = +/- 0.5%.

    Returns:
        The original DataFrame with a new 'Target_Class' column.
    """
    data = df.copy()
    
    # 1. Calculate future price and percent change
    #    We shift the price column UP by 'horizon' steps
    future_price = data[price_col].shift(-horizon)
    percent_change = (future_price - data[price_col]) / data[price_col]

    # 2. Define the classification rule
    def classify(change):
        if pd.isna(change):
            return np.nan # Will be dropped later
        
        if change > neutral_threshold:
            return 2 # 'Up'
        elif change < -neutral_threshold:
            return 0 # 'Down'
        else:
            return 1 # 'Hold'

    # 3. Apply the rule
    data['Target_Class'] = percent_change.apply(classify)
    
    return data


def create_regression_target(df: pd.DataFrame, 
                             price_col: str = 'Close', 
                             horizon: int = 1) -> pd.DataFrame:
    """
    Creates a regression target.

    The target is simply the price at the future horizon.

    Args:
        df: The input DataFrame (must be sorted by date).
        price_col: The name of the column to base the prediction on (e.g., 'Close').
        horizon: The number of days to look ahead (e.g., 1 = predict next day).

    Returns:
        The original DataFrame with a new 'Target_Price' column.
    """
    data = df.copy()
    
    # 1. Shift the price data UP by 'horizon' steps.
    #    This assigns the future price to the current row.
    data['Target_Price'] = data[price_col].shift(-horizon)
    
    return data

# --- Demonstration ----------------------------------------------------------

if __name__ == "__main__":
    
    print("--- Starting Label Creation Demo ---")

    # 1. Create a sample DataFrame
    sample_data = {
        'Date': pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
            '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'
        ]),
        'Close': [
            100.00, # Day 1
            102.00, # Day 2 (+2.0%) -> Up (2)
            102.50, # Day 3 (+0.49%) -> Hold (1)
            101.00, # Day 4 (-1.46%) -> Down (0)
            101.00, # Day 5 (0.0%) -> Hold (1)
            103.00, # Day 6 (+1.98%) -> Up (2)
            102.00, # Day 7 (-0.97%) -> Down (0)
            105.00  # Day 8
        ]
    }
    df = pd.DataFrame(sample_data).set_index('Date')
    
    print("--- Original Data ---")
    print(df)
    
    # --- Demo 1: Classification Target ---
    print("\n--- Demo 1: Classification Target (horizon=1, threshold=0.5%) ---")
    
    # We use a 0.5% threshold (0.005)
    df_class = create_classification_target(df, 
                                            price_col='Close', 
                                            horizon=1, 
                                            neutral_threshold=0.005)
    
    # Note: The last row (Day 8) will have NaN for the target,
    # as its future is unknown.
    print(df_class)

    # --- Demo 2: Regression Target ---
    print("\n--- Demo 2: Regression Target (horizon=1) ---")
    
    df_reg = create_regression_target(df, 
                                      price_col='Close', 
                                      horizon=1)
    
    print(df_reg)
    
    # --- How to use this for training ---
    print("\n--- Final DataFrame for Training (after dropping NaNs) ---")
    
    # We use the classification DF as an example
    # In a real project, you'd run this ONCE after feature engineering
    
    final_df = df_class.dropna()
    
    print(final_df)
    
    print("\n--- Demo Complete ---")
    print("You can now separate this final DataFrame into features (X) and labels (y).")
