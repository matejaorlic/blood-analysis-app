import numpy as np


def generate_normal_patient():

    patient = {
        "WBC": round(np.random.normal(7.0, 1.2), 2),
        "RBC": round(np.random.normal(4.8, 0.4), 2),
        "HGB": round(np.random.normal(14.5, 1.0), 2),
        "HCT": round(np.random.normal(43, 3), 2),
        "PLT": round(np.random.normal(280, 40), 2),
        "MCV": round(np.random.normal(90, 4), 2),
        "Neutrophils": round(np.random.normal(58, 6), 2),
        "Lymphocytes": round(np.random.normal(32, 5), 2),
        "Class": "Normal"
    }

    return patient


patient = generate_normal_patient()

print(patient)