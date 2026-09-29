import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def run_eda(input_path: str = 'data/processed/heart_processed.csv',
            output_dir: str = 'visualizations') -> None:
    """
    Performs Exploratory Data Analysis (EDA) on the processed heart disease dataset
    and saves key plots in the output directory.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed dataset not found at: {input_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    print(f"Loaded processed dataset from {input_path} with shape {df.shape}.")

    target_labels = {0: 'No Disease (0)', 1: 'Heart Disease (1)'}
    df['target_label'] = df['target'].map(target_labels)
    color_palette = {'No Disease (0)': '#2ecc71', 'Heart Disease (1)': '#e74c3c'}

    # 1. Target distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x='target_label', hue='target_label', palette=color_palette, legend=False)
    plt.title("Heart Disease Target Distribution", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Diagnosis Status", fontsize=12)
    plt.ylabel("Count of Patients", fontsize=12)
    
    # Add count annotations on top of bars
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3),
                        textcoords='offset points')
                    
    plt.tight_layout()
    target_dist_path = os.path.join(output_dir, 'target_distribution.png')
    plt.savefig(target_dist_path, dpi=300)
    plt.close()
    print(f"Saved: {target_dist_path}")

    # 2. Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='target_label', kde=True, bins=20, 
                 palette=color_palette, element="step", common_norm=False)
    plt.title("Age Distribution by Heart Disease Status", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Age (Years)", fontsize=12)
    plt.ylabel("Patient Count", fontsize=12)
    plt.tight_layout()
    age_dist_path = os.path.join(output_dir, 'age_distribution.png')
    plt.savefig(age_dist_path, dpi=300)
    plt.close()
    print(f"Saved: {age_dist_path}")

    # 3. Cholesterol vs target
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='target_label', y='chol', hue='target_label', palette=color_palette, legend=False, width=0.4)
    sns.stripplot(data=df, x='target_label', y='chol', color='black', alpha=0.3, jitter=0.2)
    plt.title("Serum Cholesterol Levels by Target Status", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Diagnosis Status", fontsize=12)
    plt.ylabel("Cholesterol (mg/dl)", fontsize=12)
    plt.tight_layout()
    chol_path = os.path.join(output_dir, 'cholesterol_vs_target.png')
    plt.savefig(chol_path, dpi=300)
    plt.close()
    print(f"Saved: {chol_path}")

    # 4. Resting blood pressure vs target
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='target_label', y='trestbps', hue='target_label', palette=color_palette, legend=False, width=0.4)
    sns.stripplot(data=df, x='target_label', y='trestbps', color='black', alpha=0.3, jitter=0.2)
    plt.title("Resting Blood Pressure by Target Status", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Diagnosis Status", fontsize=12)
    plt.ylabel("Resting Blood Pressure (mm Hg)", fontsize=12)
    plt.tight_layout()
    rbp_path = os.path.join(output_dir, 'resting_bp_vs_target.png')
    plt.savefig(rbp_path, dpi=300)
    plt.close()
    print(f"Saved: {rbp_path}")

    # 5. Maximum heart rate vs target
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='target_label', y='thalach', hue='target_label', palette=color_palette, legend=False, width=0.4)
    sns.stripplot(data=df, x='target_label', y='thalach', color='black', alpha=0.3, jitter=0.2)
    plt.title("Maximum Heart Rate Achieved (thalach) by Target Status", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Diagnosis Status", fontsize=12)
    plt.ylabel("Max Heart Rate (bpm)", fontsize=12)
    plt.tight_layout()
    hr_path = os.path.join(output_dir, 'max_heart_rate_vs_target.png')
    plt.savefig(hr_path, dpi=300)
    plt.close()
    print(f"Saved: {hr_path}")

    # 6. Correlation heatmap
    plt.figure(figsize=(12, 9))
    numeric_df = df.drop(columns=['target_label'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_path}")

    print(f"\nAll 6 EDA visualizations successfully generated in '{output_dir}/'.")

if __name__ == '__main__':
    run_eda()

