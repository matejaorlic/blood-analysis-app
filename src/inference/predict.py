import joblib

model = joblib.load("models/model.pkl")

print("Model loaded successfully.")

new_patient = [[
    13.5,   # WBC
    4.6,    # RBC
    14.1,   # HGB
    42.0,   # HCT
    320,    # PLT
    88.0,   # MCV
    76.0,   # Neutrophils
    18.0    # Lymphocytes
]]

prediction = model.predict(new_patient)

predicted_class = prediction[0]

print("\nBased on the blood analysis results,")
print(f"the patient's condition is most likely: {predicted_class}")

probabilities = model.predict_proba(new_patient)
classes = model.classes_
print("\nPrediction probabilities:")

for class_name, probability in zip(classes, probabilities[0]):

    print(f"{class_name}: {probability * 100:.2f}%")