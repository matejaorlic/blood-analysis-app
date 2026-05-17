import pandas as pd

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