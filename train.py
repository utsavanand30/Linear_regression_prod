import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


os.makedirs("models",exist_ok=True)

df=pd.read_csv("data/ads.csv")

print(df.head())
print(df.columns)

x=df.drop("Sales",axis=1)
y=df["Sales"]

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=42)

model= LinearRegression()
model.fit(train_x,train_y)

joblib.dump(model,'models/linear_regression_model.pkl')
            
