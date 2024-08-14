import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from matplotlib.ticker import FuncFormatter
import numpy as np

# Load the Dataset
df = pd.read_csv('ds_salaries.csv')
df = df.dropna()  # Drop rows with missing values
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')

# Identify categorical columns (check if they need encoding)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Convert categorical columns to category dtype
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Prepare features and target variable
X = df.drop(columns=['salary'])
y = df['salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Save model
joblib.dump(model, 'LinearRegression_model.pkl')

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression Mean Squared Error: {mse}")

# Convert y_test and y_pred to millions
y_test_millions = y_test / 1e6
y_pred_millions = y_pred / 1e6

# Calculate the slope (m) and intercept (b) for the regression line in terms of millions
m, b = np.polyfit(y_test_millions, y_pred_millions, 1)
regression_eq = f"y = {m:.2f}x + {b:.2f}"

# Plot predicted vs actual with regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_millions, y=y_pred_millions, alpha=0.5, color='blue')

# Plot the regression line as a dashed line
plt.plot(y_test_millions, m * y_test_millions + b, color='red', linestyle='--', label=regression_eq)

plt.xlabel('Actual Salary ($ Millions)')
plt.ylabel('Predicted Salary ($ Millions)')
plt.title('Linear Regression - Predicted vs Actual Salaries')
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)

# Display the regression equation on the plot
plt.text(0.2, 1.2, regression_eq, fontsize=12, color='red')

# Save the plot to the current directory
plt.savefig('LinearRegression_predicted_vs_actual_with_dashed_line.png')
plt.show()

# Ensure residuals don't have NaN values and are within a reasonable range
residuals = y_test_millions - y_pred_millions
residuals = residuals[~(residuals.isna() | (residuals == float('inf')) | (residuals == float('-inf')))]

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue', bins=30)
plt.title('Linear Regression - Residuals Distribution')
plt.xlabel('Residuals (Millions)')
plt.ylabel('Frequency')

# Save the residuals plot to the current directory
plt.savefig('LinearRegression_residuals.png')
plt.show()

# Save the MSE and results to a file in the current directory
with open('LinearRegression_results.txt', 'w') as f:
    f.write(f"Linear Regression Mean Squared Error: {mse}\n")
    f.write(f"Regression Equation: {regression_eq}\n")
