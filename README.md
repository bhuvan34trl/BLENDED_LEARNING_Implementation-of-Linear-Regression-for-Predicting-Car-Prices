# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset and select features (engine size, horsepower, mileage) and target (price).
2. plit the data into training and testing sets, then standardize the feature values.
3. Train a Linear Regression model using the training data and predict prices for the test data.
4. Evaluate the model using performance metrics (MSE, RMSE, MAE, R²) and visualize results to check model accuracy.

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Bhuvanesh.K
RegisterNumber: 212225230035 
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head()
X=df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y=df['price']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print('Name:Bhuvanesh.K')
print('Reg. No: 212225230035')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test, y_pred):>10.2f}")
print(f"{'MAE':>12}: {mean_absolute_error(y_test, y_pred):>10.2f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(),y.max()], 'r--')
plt.title("Linearity Check: Actual vs predicted prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals =y_test - y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs predicted")
plt.xlabel("Predicted price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
fig,(ax1, ax2) =plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
<img width="437" height="300" alt="Screenshot 2026-03-17 125803" src="https://github.com/user-attachments/assets/f3432aaa-d687-4755-aa29-056b07db17e9" />

<img width="1429" height="619" alt="Screenshot 2026-03-17 125814" src="https://github.com/user-attachments/assets/633093cb-7dab-4a9a-8175-2b1ee438ba3b" />

<img width="1379" height="574" alt="Screenshot 2026-03-17 125843" src="https://github.com/user-attachments/assets/734c94cf-122e-4ccb-a914-c2d9b474aceb" />

<img width="1391" height="547" alt="Screenshot 2026-03-17 125910" src="https://github.com/user-attachments/assets/8374570d-ff61-49c8-924b-93c230c80667" />





## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
