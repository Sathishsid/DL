# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='Price')

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Step 7: Plot Training Data
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 5], y_train, color='blue', label='Training Data')  # Feature: RM (Average number of rooms per dwelling)
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('House Price')
plt.title('Training Data')
plt.legend()
plt.show()

# Step 8: Plot Predictions vs. Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Diagonal line for reference
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual House Prices')
plt.legend()
plt.show()

# Step 9: Make Predictions
sample_data = np.array([[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.9, 4.98]])
sample_data = scaler.transform(sample_data)
predicted_price = model.predict(sample_data)
print(f'Predicted Price for the sample data: ${predicted_price[0]*1000:.2f}')
