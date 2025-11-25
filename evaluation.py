
"""

It includes:
1. Standard Classification Metrics (Accuracy, F1, Confusion Matrix).
2. A Simple "Equity Curve" Simulator to calculate potential profit.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    """
    Prints standard classification metrics.
    """
    print(f"\n--- Performance Report: {model_name} ---")
    
    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2%}")
    
    # 2. Detailed Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return acc

def simulate_trading_strategy(dates, prices, predictions, initial_capital=10000):
    """
    Simulates a simple trading strategy based on model predictions.
    
    Strategy:
    - If Prediction == 1 (Up): Buy/Hold Stock.
    - If Prediction == 0 (Down): Sell to Cash.
    
    Args:
        dates: List/Series of dates.
        prices: List/Series of actual stock prices (e.g., Close).
        predictions: List/Series of model predictions (0 or 1).
        initial_capital: Starting money.
        
    Returns:
        DataFrame containing the portfolio value over time.
    """
    print("\n--- Running Trading Simulation ---")
    
    # Create a DataFrame to track the simulation
    portfolio = pd.DataFrame(index=dates)
    portfolio['Price'] = prices.values
    portfolio['Signal'] = predictions # 1 = Long, 0 = Cash
    
    # Initialize variables
    cash = initial_capital
    shares = 0
    portfolio_value = []
    
    for i in range(len(portfolio)):
        price = portfolio['Price'].iloc[i]
        signal = portfolio['Signal'].iloc[i]
        
        # LOGIC:
        # If Signal is 1 and we have cash, BUY.
        if signal == 1 and cash > 0:
            shares = cash / price
            cash = 0
            
        # If Signal is 0 and we have shares, SELL.
        elif signal == 0 and shares > 0:
            cash = shares * price
            shares = 0
            
        # Calculate total value (Cash + Current Value of Shares)
        current_value = cash + (shares * price)
        portfolio_value.append(current_value)
        
    portfolio['Portfolio_Value'] = portfolio_value
    
    # Calculate Buy & Hold Strategy for comparison
    initial_shares = initial_capital / portfolio['Price'].iloc[0]
    portfolio['Buy_and_Hold'] = initial_shares * portfolio['Price']
    
    # Calculate Returns
    model_return = (portfolio['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital
    bnh_return = (portfolio['Buy_and_Hold'].iloc[-1] - initial_capital) / initial_capital
    
    print(f"Initial Capital: ${initial_capital}")
    print(f"Final Model Value: ${portfolio['Portfolio_Value'].iloc[-1]:.2f} (Return: {model_return:.2%})")
    print(f"Final Buy & Hold Value: ${portfolio['Buy_and_Hold'].iloc[-1]:.2f} (Return: {bnh_return:.2%})")
    
    return portfolio

# --- Demonstration ----------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Create Dummy Test Data
    dates = pd.date_range(start='2023-01-01', periods=20)
    # Price goes up generally, but has a dip in the middle
    prices = [100, 101, 102, 103, 104, 100, 95, 90, 92, 95, 98, 102, 105, 106, 108, 110, 112, 111, 113, 115]
    
    # Actual Targets (1 if price went up next day, else 0)
    # We create dummy targets just for the demo
    y_true = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0] # Example
    
    # Model Predictions (Let's say it's 80% accurate)
    y_pred = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    
    # 2. Evaluate Metrics
    evaluate_model_performance(y_true, y_pred, model_name="Graph-CNN Hybrid")
    
    # 3. Run Simulation
    # Note: We pass the prices to calculate money
    results = simulate_trading_strategy(dates, pd.Series(prices), y_pred)
    
    # Optional: Plotting (if you have matplotlib installed)
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(results.index, results['Portfolio_Value'], label='AI Model Strategy', color='green')
        plt.plot(results.index, results['Buy_and_Hold'], label='Buy & Hold', color='gray', linestyle='--')
        plt.title('Backtest: AI Model vs Buy & Hold')
        plt.legend()
        plt.show()
        print("Plot generated.")
    except Exception as e:
        print("Skipping plot (matplotlib issue or not interactive).")
