import numpy as np

def risk_level(nodules, diameters, cls, confidences):
    cls_normalized = str(cls).strip().lower()
    is_cancer = cls_normalized in {"cancer", "malignant"}

    if nodules == 0:
        return "High Risk" if is_cancer else "Normal"

    max_diameter = max(diameters)
    avg_conf = sum(confidences) / len(confidences)

    if is_cancer:
        return "High Risk"

    if nodules >= 5 or max_diameter > 60:
        return "High Risk"

    if nodules >= 2 or max_diameter > 30:
        return "Medium Risk"

    if avg_conf > 0.7:
        return "Medium Risk"

    return "Low Risk"
