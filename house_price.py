import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Price distribution
plt.figure(figsize=(6,4))
sns.histplot(df['price'], kde=True)
plt.title("House Price Distribution")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()
plt.close()

# Check columns
print("\nColumns:")
print(df.columns)

# Data cleaning
df.fillna(df.mean(numeric_only=True), inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\nLinear Regression Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2  :", r2_score(y_test, y_pred))

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Performance:")
print("R2 :", r2_score(y_test, rf_pred))

# Feature importance
importance = rf.feature_importances_
features = X.columns

fi = pd.Series(importance, index=features).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(fi.head(10))

plt.figure(figsize=(8,5))
fi.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
plt.close()
