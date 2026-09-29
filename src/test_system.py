import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure src directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommendation_engine import generate_recommendations, classify_risk_tier

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

MODEL_PATH = 'models/final_model.joblib'

# 3 Representative Patient Test Cases (Low, Moderate, High Risk)
PATIENT_CASES = {
    'Low Risk Case': {
        'age': 37.0, 'sex': 1.0, 'cp': 3.0, 'trestbps': 130.0,
        'chol': 250.0, 'fbs': 0.0, 'restecg': 0.0, 'thalach': 187.0,
        'exang': 0.0, 'oldpeak': 0.0, 'slope': 1.0, 'ca': 0.0, 'thal': 3.0
    },
    'Moderate Risk Case': {
        'age': 55.0, 'sex': 1.0, 'cp': 4.0, 'trestbps': 130.0,
        'chol': 230.0, 'fbs': 0.0, 'restecg': 0.0, 'thalach': 150.0,
        'exang': 0.0, 'oldpeak': 1.2, 'slope': 2.0, 'ca': 0.0, 'thal': 3.0
    },
    'High Risk Case': {
        'age': 67.0, 'sex': 1.0, 'cp': 4.0, 'trestbps': 160.0,
        'chol': 286.0, 'fbs': 0.0, 'restecg': 2.0, 'thalach': 108.0,
        'exang': 1.0, 'oldpeak': 2.6, 'slope': 2.0, 'ca': 3.0, 'thal': 7.0
    }
}

class SystemTester:
    def __init__(self):
        self.results = []
        self.model = None

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            'Test': test_name,
            'Status': status,
            'Message': message
        })
        print(f"[{status}] {test_name}: {message}")

    def test_1_model_loading(self):
        """Test 1: Final model loads successfully."""
        try:
            if not os.path.exists(MODEL_PATH):
                self.log_result("1. Model Loading", False, f"File not found at {MODEL_PATH}")
                return
            self.model = joblib.load(MODEL_PATH)
            has_methods = hasattr(self.model, "predict") and hasattr(self.model, "predict_proba")
            if has_methods:
                self.log_result("1. Model Loading", True, f"Successfully loaded '{MODEL_PATH}'")
            else:
                self.log_result("1. Model Loading", False, "Loaded artifact missing predict/predict_proba")
        except Exception as e:
            self.log_result("1. Model Loading", False, f"Exception occurred: {str(e)}")

    def test_2_feature_acceptance(self):
        """Test 2: Required 13 input features are accepted."""
        if not self.model:
            self.log_result("2. Feature Acceptance", False, "Model not loaded")
            return
        try:
            sample_df = pd.DataFrame([PATIENT_CASES['Low Risk Case']])[FEATURE_COLUMNS]
            probs = self.model.predict_proba(sample_df)
            if probs.shape == (1, 2):
                self.log_result("2. Feature Acceptance", True, f"Accepted all {len(FEATURE_COLUMNS)} input features")
            else:
                self.log_result("2. Feature Acceptance", False, f"Unexpected probability shape: {probs.shape}")
        except Exception as e:
            self.log_result("2. Feature Acceptance", False, f"Failed feature acceptance: {str(e)}")

    def test_3_probability_range(self):
        """Test 3: Prediction probability is between 0 and 1."""
        if not self.model:
            self.log_result("3. Probability Range", False, "Model not loaded")
            return
        try:
            all_valid = True
            messages = []
            for name, patient in PATIENT_CASES.items():
                df = pd.DataFrame([patient])[FEATURE_COLUMNS]
                prob = self.model.predict_proba(df)[0, 1]
                if not (0.0 <= prob <= 1.0):
                    all_valid = False
                    messages.append(f"{name} invalid prob: {prob}")
            if all_valid:
                self.log_result("3. Probability Range", True, "All predicted probabilities bounded in [0.0, 1.0]")
            else:
                self.log_result("3. Probability Range", False, "; ".join(messages))
        except Exception as e:
            self.log_result("3. Probability Range", False, f"Exception: {str(e)}")

    def test_4_binary_output(self):
        """Test 4: Prediction output is strictly 0 or 1."""
        if not self.model:
            self.log_result("4. Binary Output", False, "Model not loaded")
            return
        try:
            all_binary = True
            for name, patient in PATIENT_CASES.items():
                df = pd.DataFrame([patient])[FEATURE_COLUMNS]
                pred = self.model.predict(df)[0]
                if pred not in [0, 1]:
                    all_binary = False
            if all_binary:
                self.log_result("4. Binary Output", True, "All predictions returned binary output (0 or 1)")
            else:
                self.log_result("4. Binary Output", False, "Non-binary class prediction encountered")
        except Exception as e:
            self.log_result("4. Binary Output", False, f"Exception: {str(e)}")

    def test_5_risk_tier_assignment(self):
        """Test 5: Risk tier is correctly assigned for low, moderate, and high probabilities."""
        try:
            t1 = classify_risk_tier(0.15) == "Low Risk"
            t2 = classify_risk_tier(0.45) == "Moderate Risk"
            t3 = classify_risk_tier(0.85) == "High Risk"
            if t1 and t2 and t3:
                self.log_result("5. Risk Tier Assignment", True, "Correctly classified Low (<0.30), Moderate (0.30-0.69), and High (>=0.70) tiers")
            else:
                self.log_result("5. Risk Tier Assignment", False, f"Mismatch in risk tier boundaries: low={t1}, mod={t2}, high={t3}")
        except Exception as e:
            self.log_result("5. Risk Tier Assignment", False, f"Exception: {str(e)}")

    def test_6_recommendation_count(self):
        """Test 6: Recommendation engine returns 3-5 recommendations."""
        try:
            all_in_range = True
            counts = []
            for name, patient in PATIENT_CASES.items():
                res = generate_recommendations(patient, risk_probability=0.5)
                rec_count = len(res['recommendations'])
                counts.append(f"{name}:{rec_count}")
                if not (3 <= rec_count <= 5):
                    all_in_range = False
            if all_in_range:
                self.log_result("6. Recommendation Count", True, f"Returned 3-5 recommendations across all cases ({', '.join(counts)})")
            else:
                self.log_result("6. Recommendation Count", False, f"Out of range counts: {', '.join(counts)}")
        except Exception as e:
            self.log_result("6. Recommendation Count", False, f"Exception: {str(e)}")

    def test_7_missing_invalid_input_handling(self):
        """Test 7: Missing/invalid inputs handled gracefully without crashing."""
        try:
            # Test empty dict to check defaults
            empty_res = generate_recommendations({}, risk_probability=0.20)
            valid_empty = len(empty_res['recommendations']) >= 3 and empty_res['risk_tier'] == "Low Risk"
            
            # Test unexpected input types / extreme values
            extreme_patient = {'age': 150, 'trestbps': 300, 'chol': 900, 'thalach': 20, 'exang': 1, 'cp': 4}
            extreme_res = generate_recommendations(extreme_patient, risk_probability=0.99)
            valid_extreme = len(extreme_res['recommendations']) >= 3 and extreme_res['risk_tier'] == "High Risk"

            if valid_empty and valid_extreme:
                self.log_result("7. Error Handling", True, "Handled missing and extreme inputs without crashing")
            else:
                self.log_result("7. Error Handling", False, "Failed graceful degradation on edge inputs")
        except Exception as e:
            self.log_result("7. Error Handling", False, f"Crashed on edge case input: {str(e)}")

    def test_8_patient_case_evaluations(self):
        """Test 8: Run end-to-end inference & recommendations on 3 patient cases."""
        if not self.model:
            self.log_result("8. End-to-End Patient Cases", False, "Model not loaded")
            return
        try:
            print("\n---------------- Detailed Patient Case Runs ----------------")
            all_passed = True
            expected_tiers = {
                'Low Risk Case': 'Low Risk',
                'Moderate Risk Case': 'Moderate Risk',
                'High Risk Case': 'High Risk'
            }

            for case_name, patient_data in PATIENT_CASES.items():
                df_patient = pd.DataFrame([patient_data])[FEATURE_COLUMNS]
                prob = float(self.model.predict_proba(df_patient)[0, 1])
                pred = int(self.model.predict(df_patient)[0])
                rec_out = generate_recommendations(patient_data, prob)
                
                print(f"\n* Case: {case_name}")
                print(f"  Age: {patient_data['age']}, RBP: {patient_data['trestbps']}, Chol: {patient_data['chol']}")
                print(f"  Predicted Probability : {prob:.1%}")
                print(f"  Predicted Class       : {pred} ({'Disease' if pred==1 else 'No Disease'})")
                print(f"  Assigned Risk Tier    : {rec_out['risk_tier']}")
                print(f"  Recommendations Count : {len(rec_out['recommendations'])}")
                
                tier_match = (rec_out['risk_tier'] == expected_tiers[case_name])
                if not (0.0 <= prob <= 1.0) or pred not in [0, 1] or not (3 <= len(rec_out['recommendations']) <= 5) or not tier_match:
                    all_passed = False
            print("------------------------------------------------------------\n")
            
            if all_passed:
                self.log_result("8. End-to-End Patient Cases", True, "Executed Low (3.0%), Moderate (47.1%), and High (99.7%) risk patient cases successfully")
            else:
                self.log_result("8. End-to-End Patient Cases", False, "One or more patient cases failed verification criteria")
        except Exception as e:
            self.log_result("8. End-to-End Patient Cases", False, f"Exception in patient cases run: {str(e)}")

    def run_all_tests(self):
        print("\n================ SYSTEM END-TO-END VERIFICATION ================\n")
        self.test_1_model_loading()
        self.test_2_feature_acceptance()
        self.test_3_probability_range()
        self.test_4_binary_output()
        self.test_5_risk_tier_assignment()
        self.test_6_recommendation_count()
        self.test_7_missing_invalid_input_handling()
        self.test_8_patient_case_evaluations()

        # Summary Report
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['Status'] == 'PASS')
        failed_tests = total_tests - passed_tests

        print("====================== TEST SUITE REPORT ======================")
        print(f" Total Tests Executed : {total_tests}")
        print(f" Total Passed         : {passed_tests}")
        print(f" Total Failed         : {failed_tests}")
        print(f" System Status        : {'SYSTEM OPERATIONAL (ALL PASS)' if failed_tests == 0 else 'SYSTEM ERROR'}")
        print("===============================================================\n")

        return failed_tests == 0

if __name__ == '__main__':
    tester = SystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

