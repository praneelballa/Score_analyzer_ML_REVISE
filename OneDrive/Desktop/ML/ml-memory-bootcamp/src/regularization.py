import numpy as np
import pandas as pd

class RegularizedLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1500, lambda_reg=0.0):
        self.lr = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.w = None
        self.b = 0.0

    def compute_cost(self, X, y):
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        # MSE + L2 penalty on weights (excluding bias b)
        reg_penalty = (self.lambda_reg / (2 * m)) * np.sum(self.w ** 2)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2) + reg_penalty
        return cost

    def compute_gradient(self, X, y):
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        # Gradient with regularization penalty added to weights
        dw = (1 / m) * np.dot(X.T, (predictions - y)) + (self.lambda_reg / m) * self.w
        db = (1 / m) * np.sum(predictions - y)
        return dw, db

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for _ in range(self.iterations):
            dw, db = self.compute_gradient(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

def generate_polynomial_features(X, degree=2):
    """
    Generates polynomial features to create a high-capacity model that overfits.
    """
    m, n = X.shape
    features = [X]
    for d in range(2, degree + 1):
        features.append(X ** d)
    return np.hstack(features)

# --- Execution Example ---
if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv("../data/student_performance.csv")
    X_raw = df[['study_hours', 'attendance', 'previous_score', 'sleep']].values
    y = df['exam_score'].values

    # 2. Induce overfitting with high-degree polynomial features
    X_poly = generate_polynomial_features(X_raw, degree=3)

    # Split into Train (80%) and Test (20%)
    split_idx = int(0.8 * len(X_poly))
    X_train, X_test = X_poly[:split_idx], X_poly[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0 # prevent division by zero
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # 3. Experiment with different Lambda (regularization) values
    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"{'Lambda (λ)':<12} | {'Train MSE':<12} | {'Test MSE':<12}")
    print("-" * 42)

    for lmbda in lambdas:
        model = RegularizedLinearRegression(learning_rate=0.01, iterations=2000, lambda_reg=lmbda)
        model.fit(X_train_scaled, y_train)
        
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        train_mse = np.mean((train_preds - y_train) ** 2)
        test_mse = np.mean((test_preds - y_test) ** 2)
        
        print(f"{lmbda:<12} | {train_mse:<12.2f} | {test_mse:<12.2f}")