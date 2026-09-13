# 1. Import libraries
import pandas as p

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


# 2. Load the CSV file
df = p.read_csv("house_price_data.csv")

print(df)


# 3. Separate Features (X) and Target (y)

X = df[["Size", "Bedrooms", "Age"]]

y = df["Price"]


# 4. Split the data into Training and Testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Create the Linear Regression model

model = LinearRegression()


# 6. Train the model using Training data

model.fit(X_train, y_train)


# 7. Make predictions using Testing data

y_pred = model.predict(X_test)


# 8. Print actual vs predicted prices

print("Actual prices:")
print(y_test)

print("Predicted prices:")
print(y_pred)


# 9. Evaluate the model

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


# 10. Print evaluation results

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R² Score:", r2)


# 11. Predict a NEW house

new_house = p.DataFrame({
    "Size": [2500],
    "Bedrooms": [4],
    "Age": [4]
})

prediction = model.predict(new_house)

print("New house predicted price:", prediction)