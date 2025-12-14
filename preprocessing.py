import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN 

class DataPreprocessor:
    def preprocess_data(self, df, target_col):
        """
        Drops ID columns, encodes text, and separates X and y.
        """
        # --- CRITICAL FIX: DROP NOISY COLUMNS ---
        # These columns confuse the model. We drop them to boost accuracy.
        drop_cols = [target_col, 'index', 'Patient Id', 'Patient_Id', 'Id']
        
        # Only drop columns that actually exist in the dataframe
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=existing_drop_cols)
        y = df[target_col]
        # ----------------------------------------
        
        # Encode text columns
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
        # Encode target
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            
        return X, y

    def split_and_balance(self, X, y):
        # 1. Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 2. Apply ADASYN
        print("Applying ADASYN...")
        adasyn = ADASYN(sampling_strategy='minority', random_state=42)
        X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
        
        # 3. Scale Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train_res, y_test