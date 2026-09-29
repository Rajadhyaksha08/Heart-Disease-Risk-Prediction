import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Feature classification
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
TARGET_COLUMN = 'target'

def create_preprocessor(num_cols: list = NUMERICAL_FEATURES, 
                        cat_cols: list = CATEGORICAL_FEATURES) -> ColumnTransformer:
    """
    Creates and returns a ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    return preprocessor

def load_and_prepare_data(data_path: str = 'data/processed/heart_processed.csv', 
                          test_size: float = 0.2, 
                          random_state: int = 42):
    """
    Loads processed dataset, separates features and target, performs stratified train-test split,
    and applies preprocessing (scaling and one-hot encoding).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    
    # 1. Separate features (X) and binary target (y)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 2. Stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )

    # 3. Create ColumnTransformer preprocessor
    preprocessor = create_preprocessor()

    # 4. Fit preprocessor on X_train and transform X_train and X_test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Extract feature names post-encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + encoded_cat_cols

    print("--- ML Data Pipeline Setup Summary ---")
    print(f"Total dataset shape      : X={X.shape}, y={y.shape}")
    print(f"Train split shape (raw)  : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test split shape (raw)   : X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Preprocessed train shape : X_train_processed={X_train_processed.shape}")
    print(f"Preprocessed test shape  : X_test_processed={X_test_processed.shape}")
    print(f"Transformed Feature Count: {len(all_feature_names)}")
    print(f"Target Stratification    :")
    print(f"  y_train distribution:\n{y_train.value_counts(normalize=True).to_dict()}")
    print(f"  y_test distribution :\n{y_test.value_counts(normalize=True).to_dict()}")
    print("---------------------------------------")

    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': all_feature_names
    }

if __name__ == '__main__':
    load_and_prepare_data()

