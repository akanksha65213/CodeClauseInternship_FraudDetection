import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("creditcard.csv")

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Isolation Forest model
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.0017,  # fraud ratio
    random_state=42
)

# Train model
iso_model.fit(X_train)

# Predict (-1 = anomaly, 1 = normal)
y_pred = iso_model.predict(X_test)

# Convert predictions to match dataset labels
# anomaly (-1) -> fraud (1)
# normal (1) -> normal (0)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Evaluation
print("Confusion Matrix (Isolation Forest):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Isolation Forest):")
print(classification_report(y_test, y_pred))


