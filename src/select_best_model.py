import os
import shutil
import pandas as pd

MODEL_FILENAME_MAP = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'Random Forest': 'random_forest.joblib',
    'SVM': 'svm.joblib',
    'KNN': 'knn.joblib'
}

def select_best_model(csv_path: str = 'model_comparison.csv',
                      models_dir: str = 'models',
                      final_model_filename: str = 'final_model.joblib') -> pd.Series:
    """
    Reads model evaluation results from CSV, ranks models using the strict priority order:
    1. ROC-AUC (descending)
    2. Recall (descending)
    3. F1-Score (descending)
    4. Accuracy (descending)
    
    Copies the winning model artifact to models/final_model.joblib and prints selection details.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evaluation metrics file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Priority sorting columns
    priority_metrics = ['ROC-AUC', 'Recall', 'F1-Score', 'Accuracy']
    
    # Sort models deterministically according to user priority rules
    df_sorted = df.sort_values(by=priority_metrics, ascending=[False, False, False, False]).reset_index(drop=True)
    
    best_model = df_sorted.iloc[0]
    best_model_name = best_model['Model']
    
    # Determine source file path and target path
    source_filename = MODEL_FILENAME_MAP.get(best_model_name)
    if not source_filename:
        raise ValueError(f"Unknown model name: '{best_model_name}'. Map update required.")
        
    source_path = os.path.join(models_dir, source_filename)
    target_path = os.path.join(models_dir, final_model_filename)
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source model artifact not found at: {source_path}")

    # Copy selected model artifact to final_model.joblib
    shutil.copyfile(source_path, target_path)

    print("\n============================ BEST MODEL SELECTION ============================")
    print(f"Selected Model: {best_model_name}")
    print(f"Destination   : {target_path}\n")
    print("Metrics of Selected Model:")
    print(f"  - ROC-AUC   : {best_model['ROC-AUC']:.4f}")
    print(f"  - Recall    : {best_model['Recall']:.4f}")
    print(f"  - F1-Score  : {best_model['F1-Score']:.4f}")
    print(f"  - Accuracy  : {best_model['Accuracy']:.4f}")
    print(f"  - Precision : {best_model['Precision']:.4f}")
    
    print("\nReason for Selection:")
    print("  Models were ranked using the strict priority order:")
    print("  1. ROC-AUC  2. Recall  3. F1-Score  4. Accuracy")
    print(f"  '{best_model_name}' achieved the highest ROC-AUC score ({best_model['ROC-AUC']:.4f})")
    if len(df_sorted) > 1:
        second_best = df_sorted.iloc[1]
        print(f"  outperforming the second-ranked model ('{second_best['Model']}' with ROC-AUC {second_best['ROC-AUC']:.4f}).")
    print("==============================================================================\n")

    return best_model

if __name__ == '__main__':
    select_best_model()

