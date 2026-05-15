import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["hours_studied", "sleep_hours", "attendance"]]
y = data["score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict new data 
new_data = pd.DataFrame([[7, 8, 85]], columns=["hours_studied", "sleep_hours", "attendance"])

prediction = model.predict(new_data)

print("Predicted Score:", prediction[0])