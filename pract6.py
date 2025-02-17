import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample dataset
np.random.seed(42)
n_samples = 100

# For simple linear regression
X_simple = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 2 * X_simple + 1 + np.random.normal(0, 1, size=(n_samples, 1))

# For multiple regression, add two more features
X_multi = np.column_stack([
    X_simple,  # Original feature
    np.sin(X_simple) * 2 + np.random.normal(0, 0.5, size=(n_samples, 1)),  # Feature 2
    np.log(X_simple + 1) + np.random.normal(0, 0.5, size=(n_samples, 1))   # Feature 3
])

def perform_regression_analysis(X, y, regression_type="Simple"):
    """
    Perform regression analysis and return results
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Actual vs Predicted
    plt.subplot(131)
    plt.scatter(y_train, y_pred_train, alpha=0.5, label='Training')
    plt.scatter(y_test, y_pred_test, alpha=0.5, label='Test')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    
    # Residuals plot
    plt.subplot(132)
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    plt.scatter(y_pred_train, residuals_train, alpha=0.5, label='Training')
    plt.scatter(y_pred_test, residuals_test, alpha=0.5, label='Test')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.legend()
    
    # Residuals distribution
    plt.subplot(133)
    plt.hist(residuals_train, bins=20, alpha=0.5, label='Training')
    plt.hist(residuals_test, bins=20, alpha=0.5, label='Test')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\n{regression_type} Linear Regression Results:")
    print("=" * 50)
    print(f"Coefficients: {model.coef_.flatten()}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print("\nModel Performance:")
    print(f"R² (Training): {r2_train:.4f}")
    print(f"R² (Test): {r2_test:.4f}")
    print(f"RMSE (Training): {rmse_train:.4f}")
    print(f"RMSE (Test): {rmse_test:.4f}")
    
    return model, (r2_train, r2_test, rmse_train, rmse_test)

# Perform simple linear regression
print("Simple Linear Regression Analysis")
simple_model, simple_metrics = perform_regression_analysis(X_simple, y)

# Perform multiple linear regression
print("\nMultiple Linear Regression Analysis")
multi_model, multi_metrics = perform_regression_analysis(X_multi, y, "Multiple")

# Compare the models
print("\nModel Comparison:")
print("=" * 50)
print(f"Simple Regression R² (Test): {simple_metrics[1]:.4f}")
print(f"Multiple Regression R² (Test): {multi_metrics[1]:.4f}")
print(f"Simple Regression RMSE (Test): {simple_metrics[3]:.4f}")
print(f"Multiple Regression RMSE (Test): {multi_metrics[3]:.4f}")
