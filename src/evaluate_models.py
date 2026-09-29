import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# Ensure src directory is in Python path for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_pipeline import load_and_prepare_data

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

MODEL_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'svm': 'SVM',
    'knn': 'KNN'
}

def evaluate_models(data_path: str = 'data/processed/heart_processed.csv',
                    models_dir: str = 'models',
                    output_csv: str = 'model_comparison.csv',
                    viz_dir: str = 'visualizations') -> pd.DataFrame:
    """
    Evaluates all trained models saved in models/ on the test set from ml_pipeline.py.
    Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
    Generates comparison tables, CSV report, and visualizations.
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Load test data from ml_pipeline (same split, random_state=42)
    data_dict = load_and_prepare_data(data_path=data_path, random_state=42)
    X_test = data_dict['X_test_raw']
    y_test = data_dict['y_test']
    
    results = []
    confusion_matrices = {}

    print("\n--- Evaluating Models ---")

    for model_key, display_name in MODEL_NAMES.items():
        model_file = os.path.join(models_dir, f"{model_key}.joblib")
        if not os.path.exists(model_file):
            print(f"Warning: Model file {model_file} not found. Skipping...")
            continue
            
        # Load trained pipeline
        pipeline = joblib.load(model_file)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Probability / Confidence scores for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            y_prob = y_pred

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        confusion_matrices[display_name] = cm

        results.append({
            'Model': display_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    df_results = pd.DataFrame(results)

    # Save metrics to CSV
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved comparison metrics to: {output_csv}")

    # 2. Print clear comparison table
    print("\n================================ MODEL COMPARISON TABLE ================================")
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("========================================================================================\n")

    # 3. Generate model_comparison.png (Grouped bar plot for all metrics)
    plt.figure(figsize=(12, 6))
    df_melted = df_results.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    
    ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title("Classification Metrics Comparison Across Models", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel("Performance Score (0.0 - 1.0)", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.legend(title="Metric", loc='lower right', frameon=True)
    
    # Annotate score values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 2),
                        textcoords='offset points', rotation=45)

    plt.tight_layout()
    comparison_img_path = os.path.join(viz_dir, 'model_comparison.png')
    plt.savefig(comparison_img_path, dpi=300)
    plt.close()
    print(f"Saved: {comparison_img_path}")

    # 4. Generate individual and combined Confusion Matrix plots
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(20, 4))
    
    for idx, (model_key, display_name) in enumerate(MODEL_NAMES.items()):
        cm = confusion_matrices[display_name]
        
        # Save individual confusion matrix plot
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Disease (0)', 'Disease (1)'],
                    yticklabels=['No Disease (0)', 'Disease (1)'])
        plt.title(f"Confusion Matrix: {display_name}", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Label", fontsize=10)
        plt.ylabel("True Label", fontsize=10)
        plt.tight_layout()
        indiv_cm_path = os.path.join(viz_dir, f"cm_{model_key}.png")
        plt.savefig(indiv_cm_path, dpi=300)
        plt.close()
        print(f"Saved: {indiv_cm_path}")

        # Subplot in combined figure
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    xticklabels=['0', '1'], yticklabels=['0', '1'])
        axes[idx].set_title(display_name, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    fig.suptitle("Confusion Matrices - All Models", fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    combined_cm_path = os.path.join(viz_dir, 'confusion_matrices_all.png')
    fig.savefig(combined_cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {combined_cm_path}")

    print(f"\nAll evaluations and visualization plots successfully generated in '{viz_dir}/'.")
    return df_results

if __name__ == '__main__':
    evaluate_models()

