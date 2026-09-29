import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def analyze_feature_importance(model_path: str = 'models/final_model.joblib',
                               output_csv: str = 'feature_importance.csv',
                               output_img: str = 'visualizations/feature_importance.png') -> pd.DataFrame:
    """
    Extracts, ranks, saves, and visualizes Logistic Regression feature coefficients
    from the final model pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")

    # 1. Load final model pipeline
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # 2. Extract transformed feature names and coefficients
    raw_feature_names = preprocessor.get_feature_names_out()
    # Clean feature names for clear display (remove num__ and cat__ prefixes)
    clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in raw_feature_names]
    coefficients = classifier.coef_[0]

    # 3. Create DataFrame
    df_importance = pd.DataFrame({
        'feature': clean_feature_names,
        'coefficient': coefficients,
        'absolute_coefficient': np.abs(coefficients)
    })

    # 4. Sort by absolute_coefficient descending
    df_importance = df_importance.sort_values(by='absolute_coefficient', ascending=False).reset_index(drop=True)

    # 5. Save as feature_importance.csv
    df_importance.to_csv(output_csv, index=False)
    print(f"Saved feature importance table to: {output_csv}")

    # 6. Extract Top 15 features for visualization & printing
    top_15 = df_importance.head(15).copy()

    # 7. Print top 15 features and their coefficients
    print("\n======================= TOP 15 LOGISTIC REGRESSION FEATURES =======================")
    print(top_15[['feature', 'coefficient', 'absolute_coefficient']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("===================================================================================\n")

    # 8. Create horizontal bar chart of the top 15 features
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.figure(figsize=(10, 7))

    # Color coding: Red for positive coefficients (increases risk), Green for negative coefficients (decreases risk)
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in top_15['coefficient']]

    ax = sns.barplot(
        data=top_15,
        x='coefficient',
        y='feature',
        palette=colors,
        hue='feature',
        legend=False
    )

    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Top 15 Feature Importances (Logistic Regression Coefficients)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Coefficient Value (Impact on Heart Disease Risk)", fontsize=12)
    plt.ylabel("Transformed Feature", fontsize=12)

    # Annotate coefficient values on bars
    for p in ax.patches:
        width = p.get_width()
        y_coord = p.get_y() + p.get_height() / 2.
        offset = 0.02 if width >= 0 else -0.02
        ha = 'left' if width >= 0 else 'right'
        ax.annotate(f"{width:.4f}",
                    (width + offset, y_coord),
                    ha=ha, va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Saved horizontal bar chart visualization to: {output_img}")

    return df_importance

if __name__ == '__main__':
    analyze_feature_importance()

