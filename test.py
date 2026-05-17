
import pandas as pd
from sklearn.linear_model import LinearRegression

data= pd.read_csv("student_data.csv")

#features input
X= data[["hours_studied", "sleep_hours", "attendance"]]
y= data["score"]

#train model

model= LinearRegression()
model.fit(X,y)

#prediction

a= int(input("enter the study hours:"))
b= int(input("enter the sleep hours: "))
c= int(input("attendance:"))
new_data=pd.DataFrame([[a,b,c]], columns=["hours_studied", "sleep_hours", "attendance"])

predition= model.predict(new_data)

print("Prediction",predition[0])