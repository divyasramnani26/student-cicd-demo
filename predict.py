import pickle


# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Get student details
cgpa = float(input("Enter CGPA: "))
attendance = float(input("Enter Attendance (%): "))
coding_score = float(input("Enter Coding Score: "))
projects = int(input("Enter Number of Projects: "))
internship = int(input("Enter Internship (1 = Yes, 0 = No): "))


# Create input data
student_data = [[
    cgpa,
    attendance,
    coding_score,
    projects,
    internship
]]


# Make prediction
prediction = model.predict(student_data)[0]


# Display result
if prediction == 1:
    print("Predicted Placement: PLACED")
else:
    print("Predicted Placement: NOT PLACED")
