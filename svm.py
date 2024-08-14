import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load the Dataset
df = pd.read_csv('ds_salaries.csv')
df = df.dropna()
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Encode categorical variables
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

# SVM Model
model = SVR(C=1.0, epsilon=0.1, kernel='rbf')  # Adjusting parameters for better accuracy
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Save model
joblib.dump(model, 'SVM_model.pkl')

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"SVM Mean Squared Error: {mse}")

# Convert y_test and y_pred to millions
y_test_millions = y_test / 1e6
y_pred_millions = y_pred / 1e6

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_millions, y=y_pred_millions, alpha=0.5, color='blue')
# No diagonal line, focusing on the scatter plot itself
plt.xlabel('Actual Salary ($ Millions)')
plt.ylabel('Predicted Salary ($ Millions)')
plt.title('SVM - Predicted vs Actual Salaries')
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)
plt.savefig('SVM_predicted_vs_actual_no_line.png')
plt.show()

# Residuals
residuals = y_test_millions - y_pred_millions
residuals = residuals[~(residuals.isna() | (residuals == float('inf')) | (residuals == float('-inf')))]

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue', bins=30)
plt.title('SVM - Residuals Distribution')
plt.xlabel('Residuals (Millions)')
plt.ylabel('Frequency')
plt.savefig('SVM_residuals.png')
plt.show()

# Save the results
with open('SVM_results.txt', 'w') as f:
    f.write(f"SVM Mean Squared Error: {mse}\n")
