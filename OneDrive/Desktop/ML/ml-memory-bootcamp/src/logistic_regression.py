import numpy as np
import pandas as pd

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0.0
        self.cost_history = []

    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.
        Clips z to prevent overflow errors in exponential calculation.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """
        Computes Binary Cross-Entropy Cost J(w, b).
        """
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        a = self.sigmoid(z)
        
        # Small epsilon to prevent log(0) errors
        eps = 1e-15
        cost = -(1 / m) * np.sum(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
        return cost

    def compute_gradient(self, X, y):
        """
        Computes gradients for weights (dw) and bias (db) for logistic regression.
        """
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        a = self.sigmoid(z)
        
        dw = (1 / m) * np.dot(X.T, (a - y))
        db = (1 / m) * np.sum(a - y)
        
        return dw, db

    def fit(self, X, y):
        """
        Optimizes weights and bias using gradient descent.
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.iterations):
            dw, db = self.compute_gradient(X, y)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i:4d} | Cost: {cost:.4f}")

    def predict_prob(self, X):
        """
        Predicts pass probabilities (between 0 and 1).
        """
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predicts binary classes (0 or 1) based on a probability threshold.
        """
        return (self.predict_prob(X) >= threshold).astype(int)

# --- Execution Example ---
if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv("../data/student_performance.csv")
    
    # Create classification target: 1 if exam_score >= 60, else 0
    y = (df['exam_score'] >= 60).astype(int).values
    X = df[['study_hours', 'attendance', 'previous_score', 'sleep']].values

    # Feature scaling
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    # 2. Train logistic regression model from scratch
    print("Training Logistic Regression Model from Scratch...")
    model = LogisticRegressionFromScratch(learning_rate=0.1, iterations=1000)
    model.fit(X_scaled, y)

    # 3. Test prediction on a sample student
    # (study_hours=6, attendance=85, previous_score=75, sleep=7)
    sample_student = np.array([[6, 85, 75, 7]])
    sample_scaled = (sample_student - X_mean) / X_std
    prob = model.predict_prob(sample_scaled)[0]
    pred_class = model.predict(sample_scaled)[0]

    print(f"\nLearned Weights (w): {model.w}")
    print(f"Learned Bias (b): {model.b:.4f}")
    print(f"Pass Probability: {prob * 100:.2f}%")
    print(f"Predicted Class (1 = Pass, 0 = Fail): {pred_class}")