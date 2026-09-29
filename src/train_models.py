import os
import sys
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Filter warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure src directory is in Python path for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_pipeline import load_and_prepare_data, create_preprocessor

def train_all_models(data_path: str = 'data/processed/heart_processed.csv',
                     models_dir: str = 'models',
                     random_state: int = 42):
    """
    Trains 5 classification models using the preprocessing pipeline and saves each trained pipeline to models/.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load data and train/test splits from ml_pipeline
    data_dict = load_and_prepare_data(data_path=data_path, random_state=random_state)
    X_train = data_dict['X_train_raw']
    y_train = data_dict['y_train']

    # 2. Define the 5 classification models
    models = {
        'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier(random_state=random_state),
        'svm': SVC(random_state=random_state),
        'knn': KNeighborsClassifier()
    }

    print("\n--- Training Models ---")
    saved_paths = {}
    
    # 3. Fit each model inside a full Pipeline and save to models/
    for name, clf in models.items():
        # Combine preprocessing ColumnTransformer and classifier into a single Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', create_preprocessor()),
            ('classifier', clf)
        ])
        
        # Fit pipeline on training data
        pipeline.fit(X_train, y_train)
        
        # Save trained pipeline model
        save_path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(pipeline, save_path)
        saved_paths[name] = save_path
        
        # Print confirmation for each trained model
        print(f"[CONFIRMATION] Successfully trained and saved '{name}' model to: {save_path}")

    print("\nTraining completed for all 5 models. Models saved in 'models/'.\n")
    return saved_paths

if __name__ == '__main__':
    train_all_models()

