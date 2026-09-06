import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("data/student_placement.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)


# -----------------------------
# 2. Validate data
# -----------------------------
required_columns = [
    "CGPA",
    "Attendance",
    "CodingScore",
    "Projects",
    "Internship",
    "Placement"
]

# Check required columns
for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Missing required column: {column}")

# Check missing values
if data[required_columns].isnull().sum().sum() > 0:
    raise ValueError("Missing values detected in dataset.")

# Check that dataset is not empty
if data.empty:
    raise ValueError("Dataset is empty.")

print("Data validation passed.")


# -----------------------------
# 3. Separate features and target
# -----------------------------
X = data[
    [
        "CGPA",
        "Attendance",
        "CodingScore",
        "Projects",
        "Internship"
    ]
]

y = data["Placement"]


# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5. Train ML model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# 6. Evaluate model
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# -----------------------------
# 7. Accuracy quality check
# -----------------------------
if accuracy < 0.80:
    raise ValueError(
        f"Model accuracy is below 80%. Accuracy = {accuracy:.2f}"
    )

print("Accuracy quality check passed.")


# -----------------------------
# 8. Save trained model
# -----------------------------
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as model.pkl")