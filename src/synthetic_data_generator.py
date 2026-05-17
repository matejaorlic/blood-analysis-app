import numpy as np
import pandas as pd


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

def generate_anemia_patient():

    patient = {
        "WBC": round(np.random.normal(7.2, 1.3), 2),
        "RBC": round(np.random.normal(3.9, 0.4), 2),
        "HGB": round(np.random.normal(10.2, 1.0), 2),
        "HCT": round(np.random.normal(32, 3), 2),
        "PLT": round(np.random.normal(320, 50), 2),
        "MCV": round(np.random.normal(72, 5), 2),
        "Neutrophils": round(np.random.normal(57, 5), 2),
        "Lymphocytes": round(np.random.normal(33, 5), 2),
        "Class": "Anemia"
    }

    return patient

def generate_bacterial_patient():

    patient = {
        "WBC": round(np.random.normal(14, 2), 2),
        "RBC": round(np.random.normal(4.7, 0.4), 2),
        "HGB": round(np.random.normal(14, 1), 2),
        "HCT": round(np.random.normal(42, 3), 2),
        "PLT": round(np.random.normal(350, 60), 2),
        "MCV": round(np.random.normal(89, 4), 2),
        "Neutrophils": round(np.random.normal(78, 6), 2),
        "Lymphocytes": round(np.random.normal(18, 5), 2),
        "Class": "Bacterial"
    }

    return patient

def generate_viral_patient():

    patient = {
        "WBC": round(np.random.normal(9, 1.5), 2),
        "RBC": round(np.random.normal(4.8, 0.4), 2),
        "HGB": round(np.random.normal(14.3, 1), 2),
        "HCT": round(np.random.normal(43, 3), 2),
        "PLT": round(np.random.normal(270, 50), 2),
        "MCV": round(np.random.normal(89, 4), 2),
        "Neutrophils": round(np.random.normal(42, 6), 2),
        "Lymphocytes": round(np.random.normal(48, 6), 2),
        "Class": "Viral"
    }

    return patient

patients = []

for _ in range(100):
    patients.append(generate_normal_patient())

for _ in range(100):
    patients.append(generate_anemia_patient())

for _ in range(100):
    patients.append(generate_bacterial_patient())

for _ in range(100):
    patients.append(generate_viral_patient())

df = pd.DataFrame(patients)
print(df.head())

df.to_csv(
    "data/raw/blood_analysis_dataset.csv",
    index=False
)

print("\nDataset saved successfully.")