import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure src directory is in Python path for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommendation_engine import generate_recommendations

FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

MODEL_PATH = 'models/final_model.joblib'

def display_banner():
    print("\n" + "=" * 75)
    print("      HEART DISEASE RISK PREDICTION & RECOMMENDATION SYSTEM      ")
    print("=" * 75)
    print(" This tool uses a trained machine learning model to estimate heart")
    print(" disease risk probability and provide tailored health recommendations.")
    print("=" * 75 + "\n")

def get_numeric_input(prompt: str, min_val: float = None, max_val: float = None) -> float:
    """Prompt user for a numeric value with bounds checking."""
    while True:
        try:
            val_str = input(prompt).strip()
            val = float(val_str)
            if min_val is not None and val < min_val:
                print(f"  [!] Error: Value must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Error: Value must be at most {max_val}.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid input. Please enter a numeric value.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting program...")
            sys.exit(0)

def get_choice_input(prompt: str, options_map: dict) -> float:
    """Prompt user to select from a menu of valid categorical choices."""
    options_desc = " / ".join([f"{k}:{v}" for k, v in options_map.items()])
    full_prompt = f"{prompt} ({options_desc}): "
    while True:
        try:
            val_str = input(full_prompt).strip()
            val = float(val_str)
            if val in options_map or int(val) in options_map:
                return float(val)
            print(f"  [!] Invalid choice. Valid options are: {list(options_map.keys())}")
        except ValueError:
            print("  [!] Invalid input. Please enter one of the specified numbers.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting program...")
            sys.exit(0)

def collect_patient_data() -> dict:
    """Collects and validates all 13 patient features via terminal input."""
    print("Please provide patient clinical details:\n")
    data = {}

    data['age'] = get_numeric_input("1. Age (years): ", min_val=1, max_val=120)
    
    data['sex'] = get_choice_input("2. Sex", {0: "Female", 1: "Male"})
    
    data['cp'] = get_choice_input(
        "3. Chest Pain Type",
        {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
    )
    
    data['trestbps'] = get_numeric_input("4. Resting Blood Pressure (mm Hg): ", min_val=50, max_val=260)
    
    data['chol'] = get_numeric_input("5. Serum Cholesterol (mg/dl): ", min_val=80, max_val=700)
    
    data['fbs'] = get_choice_input("6. Fasting Blood Sugar > 120 mg/dl", {0: "False/No", 1: "True/Yes"})
    
    data['restecg'] = get_choice_input(
        "7. Resting ECG Results",
        {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
    )
    
    data['thalach'] = get_numeric_input("8. Maximum Heart Rate Achieved (bpm): ", min_val=50, max_val=230)
    
    data['exang'] = get_choice_input("9. Exercise-Induced Angina", {0: "No", 1: "Yes"})
    
    data['oldpeak'] = get_numeric_input("10. ST Depression (oldpeak): ", min_val=0.0, max_val=10.0)
    
    data['slope'] = get_choice_input(
        "11. Slope of Peak Exercise ST Segment",
        {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
    )
    
    data['ca'] = get_choice_input(
        "12. Number of Major Vessels Colored by Fluoroscopy",
        {0: "0", 1: "1", 2: "2", 3: "3"}
    )
    
    data['thal'] = get_choice_input(
        "13. Thalassemia",
        {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}
    )

    return data

def run_prediction_pipeline(patient_data: dict, model_pipeline) -> None:
    """Predicts risk probability, evaluates risk tier, and prints recommendations."""
    # Create pandas DataFrame with correct feature names & column order
    df_patient = pd.DataFrame([patient_data])[FEATURE_ORDER]

    # Predict risk probability
    risk_prob = float(model_pipeline.predict_proba(df_patient)[0, 1])
    pred_class = int(model_pipeline.predict(df_patient)[0])

    # Call recommendation engine
    rec_results = generate_recommendations(patient_data, risk_prob)

    # Display Results
    print("\n" + "=" * 75)
    print("                         PREDICTION RESULTS                         ")
    print("=" * 75)
    print(f" Predicted Risk Probability : {risk_prob:.1%} ({risk_prob:.4f})")
    print(f" Assigned Risk Tier        : {rec_results['risk_tier'].upper()}")
    print(f" Prediction Outcome        : {'POSITIVE (Heart Disease Detected)' if pred_class == 1 else 'NEGATIVE (No Heart Disease Detected)'}")
    print("-" * 75)
    print("\nPersonalized Recommendations:")
    for idx, rec in enumerate(rec_results['recommendations'], 1):
        print(f"  {idx}. {rec}")
    print("\n" + "-" * 75)
    print(f"DISCLAIMER: {rec_results['disclaimer']}")
    print("=" * 75 + "\n")

def main():
    display_banner()

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model artifact not found at '{MODEL_PATH}'.")
        print("    Please run 'src/train_models.py' and 'src/select_best_model.py' first.")
        sys.exit(1)

    print(f"Loading final trained model pipeline from '{MODEL_PATH}'...")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.\n")

    while True:
        print("Choose an option:")
        print("  [1] Make a new patient prediction")
        print("  [2] Exit application")
        
        try:
            choice = input("\nEnter choice (1 or 2): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting program...")
            break

        if choice == '1':
            patient_data = collect_patient_data()
            run_prediction_pipeline(patient_data, model_pipeline)
        elif choice == '2':
            print("\nThank you for using the Heart Disease Risk Prediction System. Goodbye!\n")
            break
        else:
            print("  [!] Invalid choice. Please enter 1 or 2.\n")

if __name__ == '__main__':
    main()

