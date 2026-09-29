import os
from typing import Dict, List, Any, Union

DISCLAIMER = "This is an educational risk prediction system, not a medical diagnosis. Consult a qualified healthcare professional for medical advice."

def classify_risk_tier(risk_probability: float) -> str:
    """
    Classifies risk probability into Low, Moderate, or High Risk tiers.
    - Low Risk: < 0.30
    - Moderate Risk: 0.30 - 0.69
    - High Risk: >= 0.70
    """
    if risk_probability < 0.30:
        return "Low Risk"
    elif risk_probability <= 0.69:
        return "Moderate Risk"
    else:
        return "High Risk"

def generate_recommendations(patient_data: Dict[str, Any], risk_probability: float) -> Dict[str, Union[str, float, List[str]]]:
    """
    Generates 3-5 concise, personalized health recommendations based on patient clinical features
    and predicted risk probability.

    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient features (age, trestbps, chol, thalach, exang, cp, etc.)
    risk_probability : float
        Predicted probability of heart disease (0.0 to 1.0)

    Returns:
    --------
    dict with keys:
        - risk_probability: float
        - risk_tier: str
        - recommendations: List[str]
        - disclaimer: str
    """
    risk_tier = classify_risk_tier(risk_probability)
    recommendations: List[str] = []

    # Extract patient features with safe defaults
    age = float(patient_data.get('age', 50))
    trestbps = float(patient_data.get('trestbps', 120))
    chol = float(patient_data.get('chol', 200))
    thalach = float(patient_data.get('thalach', 150))
    exang = float(patient_data.get('exang', 0))
    cp = float(patient_data.get('cp', 1))

    # 1. Tier-Specific Primary Action Recommendation
    if risk_tier == "High Risk":
        recommendations.append(
            f"Urgent Medical Evaluation: High predicted risk ({risk_probability:.1%}). "
            "Schedule a comprehensive cardiovascular diagnostic evaluation with a cardiologist."
        )
    elif risk_tier == "Moderate Risk":
        recommendations.append(
            f"Preventative Medical Review: Moderate predicted risk ({risk_probability:.1%}). "
            "Consult a physician to review risk factors and implement a preventative monitoring plan."
        )
    else:
        recommendations.append(
            f"Wellness Maintenance: Low predicted risk ({risk_probability:.1%}). "
            "Continue practicing healthy habits and maintain routine annual health checkups."
        )

    # 2. Resting Blood Pressure Rule (trestbps)
    if trestbps >= 140:
        recommendations.append(
            f"Blood Pressure Management: Resting blood pressure is high ({trestbps:.0f} mm Hg). "
            "Reduce sodium intake, manage daily stress, and monitor blood pressure regularly."
        )
    elif trestbps >= 130:
        recommendations.append(
            f"Blood Pressure Caution: Resting blood pressure is elevated ({trestbps:.0f} mm Hg). "
            "Adopt dietary modifications (e.g., DASH diet) and limit processed foods."
        )

    # 3. Cholesterol Rule (chol)
    if chol >= 240:
        recommendations.append(
            f"Lipid Profile Attention: Serum cholesterol is high ({chol:.0f} mg/dl). "
            "Increase soluble fiber, lower saturated fat consumption, and discuss a lipid panel with your physician."
        )
    elif chol >= 200:
        recommendations.append(
            f"Cholesterol Awareness: Serum cholesterol is borderline elevated ({chol:.0f} mg/dl). "
            "Focus on heart-healthy fats (olive oil, nuts) and regular physical activity."
        )

    # 4. Exercise Angina & Heart Rate Rules (exang & thalach)
    if exang == 1.0:
        recommendations.append(
            "Exercise Safety Precaution: Exercise-induced angina was reported. "
            "Avoid unmonitored strenuous physical exertion and consult a specialist before beginning new exercise programs."
        )
    elif thalach < (220 - age) * 0.65:
        recommendations.append(
            f"Cardiovascular Fitness: Peak heart rate during exercise ({thalach:.0f} bpm) was low for age {age:.0f}. "
            "Gradually build aerobic endurance with low-impact cardio activities."
        )

    # 5. Chest Pain Type Rule (cp)
    if cp == 4.0:
        recommendations.append(
            "Asymptomatic Screening: Asymptomatic presentation (Type 4 chest pain) detected. "
            "Regular screening is vital because heart conditions can develop without obvious symptoms."
        )

    # Ensure recommendations list is concise (3 to 5 items)
    if len(recommendations) < 3:
        recommendations.append(
            "Diet & Exercise Alignment: Engage in at least 150 minutes of moderate aerobic exercise per week and follow a Mediterranean-style diet."
        )

    # Cap at top 5 recommendations
    recommendations = recommendations[:5]

    return {
        'risk_probability': risk_probability,
        'risk_tier': risk_tier,
        'recommendations': recommendations,
        'disclaimer': DISCLAIMER
    }

def print_recommendations_summary(res: Dict[str, Any]) -> None:
    """
    Utility function to display formatted recommendation results.
    """
    print("\n======================= PATIENT RISK & RECOMMENDATIONS =======================")
    print(f"Predicted Risk Probability: {res['risk_probability']:.1%}")
    print(f"Assigned Risk Tier        : {res['risk_tier']}")
    print("\nPersonalized Recommendations:")
    for idx, rec in enumerate(res['recommendations'], 1):
        print(f"  {idx}. {rec}")
    print(f"\nDISCLAIMER: {res['disclaimer']}")
    print("==============================================================================\n")

if __name__ == '__main__':
    # Test case 1: High Risk Patient
    sample_high_risk = {
        'age': 67.0, 'sex': 1.0, 'cp': 4.0, 'trestbps': 160.0,
        'chol': 286.0, 'fbs': 0.0, 'restecg': 2.0, 'thalach': 108.0,
        'exang': 1.0, 'oldpeak': 1.5, 'slope': 2.0, 'ca': 3.0, 'thal': 3.0
    }
    print("--- Test Case 1: High Risk Patient ---")
    res1 = generate_recommendations(sample_high_risk, risk_probability=0.85)
    print_recommendations_summary(res1)

    # Test case 2: Low Risk Patient
    sample_low_risk = {
        'age': 41.0, 'sex': 0.0, 'cp': 2.0, 'trestbps': 130.0,
        'chol': 204.0, 'fbs': 0.0, 'restecg': 2.0, 'thalach': 172.0,
        'exang': 0.0, 'oldpeak': 1.4, 'slope': 1.0, 'ca': 0.0, 'thal': 3.0
    }
    print("--- Test Case 2: Low Risk Patient ---")
    res2 = generate_recommendations(sample_low_risk, risk_probability=0.12)
    print_recommendations_summary(res2)

