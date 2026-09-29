# ❤️ Heart Disease Risk Prediction & Personalized Recommendation System

An AI/ML-based healthcare risk-screening project that uses patient clinical data to estimate the probability of heart disease and provide personalized, general health recommendations.

> **Disclaimer:** This project is developed for educational and academic purposes. It is **not a medical diagnostic system** and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Early identification of potential risk factors can help people understand when further medical attention may be appropriate.

This project develops a complete machine learning pipeline that:

* Preprocesses real-world heart disease data
* Performs exploratory data analysis (EDA)
* Trains and compares multiple machine learning algorithms
* Evaluates models using multiple performance metrics
* Selects a final model based on predefined evaluation criteria
* Interprets important model features
* Estimates an individual's heart disease risk probability
* Classifies the result into Low, Moderate, or High Risk
* Generates personalized, general health recommendations
* Provides an interactive terminal-based prediction system

---

## 🎯 Objectives

1. Build a complete end-to-end machine learning solution for heart disease risk prediction.
2. Perform data preprocessing and handle missing values.
3. Analyze relationships between clinical features and the target variable.
4. Train multiple classification algorithms.
5. Compare models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
6. Select a final model using a predefined metric-priority strategy.
7. Analyze model coefficients to understand influential features.
8. Develop a rule-based personalized recommendation engine.
9. Build a user-friendly terminal application.
10. Test the complete system using automated end-to-end tests.

---

## 📊 Dataset

The project uses the **UCI Heart Disease Dataset — Cleveland processed dataset**.

### Dataset characteristics

* **Records:** 303
* **Input features:** 13
* **Target:** Binary heart disease risk classification

The original dataset contains target values from `0` to `4`.

For this project:

```text
0     → 0 (No disease label)
1–4   → 1 (Disease label)
```

The dataset contains clinical attributes such as:

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `age`      | Age in years                      |
| `sex`      | Sex                               |
| `cp`       | Chest pain type                   |
| `trestbps` | Resting blood pressure            |
| `chol`     | Serum cholesterol                 |
| `fbs`      | Fasting blood sugar               |
| `restecg`  | Resting ECG results               |
| `thalach`  | Maximum heart rate achieved       |
| `exang`    | Exercise-induced angina           |
| `oldpeak`  | ST depression                     |
| `slope`    | Slope of peak exercise ST segment |
| `ca`       | Number of major vessels           |
| `thal`     | Thalassemia category              |

---

## 🏗️ Machine Learning Workflow

```text
             UCI Heart Disease Dataset
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
             Exploratory Data Analysis
                       │
                       ▼
              Train/Test Split
                       │
                       ▼
             Feature Transformation
                       │
                       ▼
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Logistic        Decision       Random
   Regression        Tree          Forest
        │              │              │
        └──────────────┼──────────────┘
                       │
                 SVM + KNN
                       │
                       ▼
               Model Evaluation
                       │
                       ▼
              Final Model Selection
                       │
                       ▼
              Feature Interpretation
                       │
                       ▼
               Risk Prediction
                       │
                       ▼
             Risk Classification
                       │
                       ▼
          Personalized Recommendations
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline performs the following operations:

* Assigns column names to the raw dataset.
* Converts `?` values into missing values.
* Converts numerical fields into appropriate numeric types.
* Handles missing values using mode imputation.
* Checks for duplicate records.
* Converts the original target into a binary classification target.
* Separates input features and target.
* Uses an 80/20 stratified train-test split.
* Standardizes numerical features using `StandardScaler`.
* Encodes categorical features using `OneHotEncoder`.

The preprocessing pipeline is included inside the trained model pipeline so that the same transformations are automatically applied during prediction.

---

## 📈 Exploratory Data Analysis

The project generates visualizations including:

* Target distribution
* Age distribution
* Cholesterol vs target
* Resting blood pressure vs target
* Maximum heart rate vs target
* Correlation heatmap

Generated visualizations are stored in:

```text
visualizations/
```

---

## 🤖 Machine Learning Models

Five classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

---

## 📊 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

### Results

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression | **0.8852** |     0.8387 | **0.9286** | **0.8814** | **0.9665** |
| Decision Tree       |     0.7377 |     0.6765 |     0.8214 |     0.7419 |     0.7440 |
| Random Forest       |     0.8689 |     0.8125 | **0.9286** |     0.8667 |     0.9426 |
| SVM                 | **0.8852** |     0.8387 | **0.9286** | **0.8814** |     0.9643 |
| KNN                 | **0.8852** | **0.8621** |     0.8929 |     0.8772 |     0.9529 |

### Final Model

**Logistic Regression** was selected as the final model using the predefined priority:

```text
1. ROC-AUC
2. Recall
3. F1-score
4. Accuracy
```

Logistic Regression achieved the highest ROC-AUC among the evaluated models:

```text
ROC-AUC : 0.9665
Recall  : 0.9286
F1      : 0.8814
Accuracy: 0.8852
```

The final trained pipeline is stored at:

```text
models/final_model.joblib
```

---

## 🔍 Feature Interpretation

Logistic Regression coefficients were analyzed to understand which transformed features had the largest influence on the model output.

The analysis includes:

* Feature name
* Coefficient
* Absolute coefficient magnitude

The complete feature analysis is available in:

```text
feature_importance.csv
```

Visualization:

```text
visualizations/feature_importance.png
```

> **Important:** Feature coefficients represent associations learned by the model from this dataset. They should not be interpreted as proof that an individual feature independently causes heart disease.

---

## 🧠 Risk Classification

The prediction probability is converted into three risk tiers:

| Probability | Risk Tier     |
| ----------- | ------------- |
| `< 30%`     | Low Risk      |
| `30% – 69%` | Moderate Risk |
| `≥ 70%`     | High Risk     |

The system uses the predicted probability generated by the final Logistic Regression pipeline.

---

## 💡 Personalized Recommendation Engine

The recommendation engine uses the predicted risk probability and relevant patient features to generate general educational recommendations.

It considers factors such as:

* Resting blood pressure
* Cholesterol
* Exercise-induced angina
* Maximum heart rate
* Chest pain category
* Overall predicted risk level

The recommendation engine is implemented in:

```text
src/recommendation_engine.py
```

Smoking-based recommendations are not included because smoking information is not present in the selected dataset.

---

## 💻 Terminal Application

The main application is:

```text
src/predict.py
```

It provides an interactive terminal interface where the user enters the 13 required clinical features.

The application then:

```text
Patient Input
      ↓
Preprocessing
      ↓
ML Model
      ↓
Risk Probability
      ↓
Risk Tier
      ↓
Personalized Recommendations
```

The application also supports multiple predictions during a single execution.

---

## 🧪 System Testing

An automated end-to-end test suite is available at:

```text
src/test_system.py
```

### Test Results

```text
Total Tests Executed : 8
Total Passed         : 8
Total Failed         : 0

System Status : SYSTEM OPERATIONAL
```

The tests verify:

* Model loading
* Feature acceptance
* Probability range
* Binary prediction output
* Risk tier classification
* Recommendation generation
* Error handling
* Low, moderate, and high-risk test cases

---

## 📁 Project Structure

```text
Heart-Disease-Risk-Prediction/
│
├── data/
│   ├── raw/
│   │   └── processed.cleveland.data
│   │
│   └── processed/
│       └── heart_processed.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── knn.joblib
│   └── final_model.joblib
│
├── visualizations/
│   ├── target_distribution.png
│   ├── age_distribution.png
│   ├── cholesterol_vs_target.png
│   ├── resting_bp_vs_target.png
│   ├── max_heart_rate_vs_target.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── confusion_matrices_all.png
│   └── feature_importance.png
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── ml_pipeline.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── select_best_model.py
│   ├── feature_analysis.py
│   ├── recommendation_engine.py
│   ├── predict.py
│   └── test_system.py
│
├── feature_importance.csv
├── model_comparison.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rajadhyaksha08/Heart-Disease-Risk-Prediction.git
cd Heart-Disease-Risk-Prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the project root:

```bash
python src/predict.py
```

The terminal application will ask for the required patient clinical information and display the predicted risk probability, risk tier, and general recommendations.

---

## 🧪 Run System Tests

```bash
python src/test_system.py
```

A successful run should report:

```text
Total Tests Executed : 8
Total Passed         : 8
Total Failed         : 0
System Status        : SYSTEM OPERATIONAL
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* SciPy

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

### Development Environment

* Visual Studio Code
* Git
* GitHub

---

## ⚠️ Limitations

This project has several limitations:

1. The model is trained on a relatively small dataset of 303 records.
2. The dataset represents a specific clinical population and may not generalize to all populations.
3. The model has not undergone clinical validation.
4. The predicted probability should not be interpreted as an actual medical diagnosis.
5. The recommendation engine uses predefined rules rather than a clinically validated recommendation framework.
6. The system does not include several potentially relevant factors such as smoking history, family history, BMI, diabetes history, and medication information.
7. Model performance on this dataset does not guarantee equivalent performance on real-world patients.

---

## 🚀 Future Scope

Possible future improvements include:

* Testing on larger and more diverse datasets.
* External validation using independent datasets.
* Hyperparameter optimization.
* Cross-validation and calibration analysis.
* Adding additional patient health features.
* Developing a web or mobile interface.
* Adding explainable AI techniques such as SHAP.
* Improving recommendation personalization.
* Adding secure patient data storage.
* Conducting proper clinical validation before any real-world medical application.

---

## 👨‍💻 Authors

**Mandar Rajadhyaksha**

Third Year Engineering Student
Datta Meghe College of Engineering, Navi Mumbai

---

## 📚 Dataset Source

UCI Machine Learning Repository — Heart Disease Dataset.

https://archive.ics.uci.edu/dataset/45/heart+disease

---

## ⚕️ Medical Disclaimer

This project is intended strictly for **educational and academic purposes**.

The predictions generated by this system are machine-learning estimates based on the selected dataset and should **not** be considered medical diagnoses, treatment recommendations, or professional medical advice.

Anyone concerned about their health should consult a qualified healthcare professional.
