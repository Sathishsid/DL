import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Generate synthetic data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Define polynomial degrees to test
degrees = [1, 4, 15]
# Plot data and models
plt.figure(figsize=(14, 5))
for i, degree in enumerate(degrees):
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    # Train the model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    # Predict and evaluate
    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)   
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, color='blue', label='Data')
    plt.scatter(X_test, y_test, color='red', label='Test Data')
    X_range = np.linspace(0, 5, 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, color='green', label=f'Poly Degree {degree}')
    plt.title(f'Degree {degree}\nTrain MSE: {train_mse:.2f}\nTest MSE: {test_mse:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
plt.tight_layout()
plt.show()
