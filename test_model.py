import os
import pickle
import pandas as pd


# -----------------------------
# Test 1: Check dataset
# -----------------------------
def test_dataset_exists():
    assert os.path.exists("data/student_placement.csv")


# -----------------------------
# Test 2: Check dataset columns
# -----------------------------
def test_dataset_columns():
    data = pd.read_csv("data/student_placement.csv")

    required_columns = [
        "CGPA",
        "Attendance",
        "CodingScore",
        "Projects",
        "Internship",
        "Placement"
    ]

    for column in required_columns:
        assert column in data.columns


# -----------------------------
# Test 3: Check for missing data
# -----------------------------
def test_no_missing_values():
    data = pd.read_csv("data/student_placement.csv")

    assert data.isnull().sum().sum() == 0


# -----------------------------
# Test 4: Check model file
# -----------------------------
def test_model_exists():
    assert os.path.exists("model.pkl")


# -----------------------------
# Test 5: Check model prediction
# -----------------------------
def test_model_prediction():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    student = [[8.5, 92, 85, 3, 1]]

    prediction = model.predict(student)

    assert prediction[0] == 99
