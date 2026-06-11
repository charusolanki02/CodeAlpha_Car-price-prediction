# ==========================================
# CAR PRICE PREDICTION USING MACHINE LEARNING
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("car data.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\nDATASET SHAPE:")
print(df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nCOLUMN NAMES:")
print(df.columns)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Car Age
df["Car_Age"] = 2025 - df["Year"]

# Remove Year column
df.drop("Year", axis=1, inplace=True)

# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL TRAINING
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED!")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("=" * 40)

print("Mean Absolute Error :", round(mae, 2))
print("Mean Squared Error  :", round(mse, 2))
print("Root Mean Squared Error :", round(rmse, 2))
print("R2 Score :", round(r2, 4))

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)

# ==========================================
# SAMPLE PREDICTION
# ==========================================

sample_car = X.iloc[0:1]

predicted_price = model.predict(sample_car)

print("\nPredicted Car Price:")
print(round(predicted_price[0], 2), "Lakhs")