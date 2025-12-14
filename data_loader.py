import pandas as pd
import os

def load_dataset(filepath):
    """
    Loads the dataset and limits it to 20,000 rows for speed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # --- CRITICAL FIX FOR SPEED ---
    # The original dataset has 600k+ rows. 
    # We restrict it to 20k to make training fast (matches your 10k+ claim).
    MAX_ROWS = 20000
    if len(df) > MAX_ROWS:
        print(f"Dataset is huge ({len(df)} rows). Sampling {MAX_ROWS} random rows...")
        df = df.sample(n=MAX_ROWS, random_state=42)
    # ------------------------------
    
    # Basic cleanup
    df = df.drop_duplicates()
    
    print(f"Data Loaded Successfully. Shape: {df.shape}")
    return df