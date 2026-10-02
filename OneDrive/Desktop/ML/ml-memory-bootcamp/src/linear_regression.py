import numpy as np
import pandas as pd

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0.0
        self.cost_history = []

    def compute_cost(self, X, y):
        """
        Computes Mean Squared Error (MSE) cost J(w, b).
        """
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def compute_gradient(self, X, y):
        """
        Computes gradients for weights (dw) and bias (db).
        """
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        
        # Gradients calculation
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        return dw, db

    def fit(self, X, y):
        """
        Runs gradient descent to optimize weights and bias.
        """
        m, n = X.shape
        # Initialize weights to zeros (or small random numbers)
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.iterations):
            dw, db = self.compute_gradient(X, y)
            
            # Update parameters using gradient descent rule
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Track cost every 100 iterations
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i:4d} | Cost: {cost:.4f}")

    def predict(self, X):
        """
        Predicts target values for given input features X.
        """
        return np.dot(X, self.w) + self.b

# --- Execution Example ---
if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv("../data/student_performance.csv")
    
    # Features and Target
    X = df[['study_hours', 'attendance', 'previous_score', 'sleep']].values
    y = df['exam_score'].values

    # Feature scaling (crucial for gradient descent convergence)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    # 2. Train model from scratch
    print("Training Linear Regression Model from Scratch...")
    model = LinearRegressionFromScratch(learning_rate=0.1, iterations=500)
    model.fit(X_scaled, y)

    # 3. Test prediction on a sample student
    # (study_hours=6, attendance=85, previous_score=75, sleep=7)
    sample_student = np.array([[6, 85, 75, 7]])
    sample_scaled = (sample_student - X_mean) / X_std
    pred_score = model.predict(sample_scaled)[0]

    print(f"\nLearned Weights (w): {model.w}")
    print(f"Learned Bias (b): {model.b:.4f}")
    print(f"Predicted Exam Score for sample student: {pred_score:.2f}")