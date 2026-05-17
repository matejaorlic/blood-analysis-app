import pandas as pd
import joblib

df = pd.read_csv("data/raw/blood_analysis_dataset.csv")

print(df.head())

X = df.drop("Class", axis=1)
y = df["Class"]

print(X.head())
print()
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
print("Model trained successfully.")

predictions = model.predict(X_test)
print(predictions[:10])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)

print(cm)


joblib.dump(
    model,
    "models/model.pkl"
)

print("Model saved successfully.")

