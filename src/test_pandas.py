import pandas as pd

data = {
    "Patient": ["John", "Anna", "Mark"],
    "Hemoglobin": [14.2, 11.8, 13.5],
    "WBC": [6.1, 8.4, 7.0]
}

df = pd.DataFrame(data)

print(df)

print("\nColumns:")
print(df.columns)

print("\nAverage hemoglobin:")
print(df["Hemoglobin"].mean())

print("\nPatients with hemoglobin below 13:")
print(df[df["Hemoglobin"] < 13])