

"""
It includes:
1. Data Preparation (Merging Time-Series and Graph Features).
2. Train/Test Splitting (Time-aware).
3. Model 1: Random Forest (sklearn).
4. Model 2: 1D-CNN (TensorFlow/Keras).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not found. DL model will be skipped.")
    TF_AVAILABLE = False

# --- 1. Data Preparation Helper ---------------------------------------------

def prepare_combined_dataset(price_features_df, graph_features_df, target_col='Target_Class'):
    """
    Merges the time-series features with the static graph features.
    """
    print("Merging datasets...")
    

    # 1. Create a copy to avoid modifying original
    combined_df = price_features_df.copy()
    
    # 2. Add graph features as new columns
    
    for feature_name, feature_value in graph_features_df.items():
        combined_df[feature_name] = feature_value

    combined_df = combined_df.dropna()
    
 
    y = combined_df[target_col]
    X = combined_df.drop(columns=[target_col])
    
    return X, y

def time_series_split(X, y, test_size=0.2):
 
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n--- Training Hybrid Random Forest ---")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    

    rf_model.fit(X_train, y_train)
    

    predictions = rf_model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Random Forest Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
   
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    print("Top 5 Important Features:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {imp:.4f}")
        
    return rf_model



def train_cnn_model(X_train, y_train, X_test, y_test, window_size=10):
    if not TF_AVAILABLE:
        return None
        
    print("\n--- Training Hybrid 1D-CNN ---")
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    def create_sequences(data, labels, window):
        X_seq, y_seq = [], []
        for i in range(len(data) - window):
            X_seq.append(data[i:i+window])
            y_seq.append(labels.iloc[i+window])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, window_size)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, window_size)
    
    print(f"CNN Input Shape: {X_train_seq.shape}")
 
    model = Sequential()
    # Conv Layer: extracts features
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_train.shape[1])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
   
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2)) # Prevent overfitting

    model.add(Dense(1, activation='sigmoid'))
    

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=1)
    

    loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"CNN Accuracy: {acc:.4f}")
    
    return model



if __name__ == "__main__":
    print("--- Starting Model Training Demo ---")
    
    # 1. Create Dummy Data
    # 100 days of data
    dates = pd.date_range(start='2023-01-01', periods=100)
    price_features = pd.DataFrame({
        'RSI': np.random.uniform(30, 70, 100),
        'SMA_50': np.random.uniform(100, 150, 100),
        'Target_Class': np.random.randint(0, 2, 100) # Binary 0 or 1
    }, index=dates)
    
  
    graph_features = {
        'CORR_degree': 0.85,
        'CORR_pagerank': 0.12,
        'NEWS_centrality': 0.45
    }
    

    X, y = prepare_combined_dataset(price_features, graph_features, 'Target_Class')
    

    X_train, X_test, y_train, y_test = time_series_split(X, y)

    rf = train_random_forest(X_train, y_train, X_test, y_test)
    
    cnn = train_cnn_model(X_train, y_train, X_test, y_test, window_size=5)
