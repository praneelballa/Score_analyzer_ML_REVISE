import numpy as np
import pandas as pd

# --- Metrics from Scratch ---

def compute_regression_metrics(y_true, y_pred):
    """
    Computes MAE, MSE, and RMSE for regression models.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def compute_classification_metrics(y_true, y_pred):
    """
    Computes Accuracy, Precision, Recall, and F1-score for binary classification.
    """
    accuracy = np.mean(y_true == y_pred)
    
    # True Positives, False Positives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1

# --- Dataset Splitting Utility ---

def split_data(X, y, train_ratio=0.70, val_ratio=0.15):
    """
    Splits data into Train, Validation, and Test sets sequentially or randomly.
    """
    m = len(X)
    indices = np.arange(m)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    train_end = int(train_ratio * m)
    val_end = int((train_ratio + val_ratio) * m)
    
    X_train, y_train = X_shuffled[:train_end], y_shuffled[:train_end]
    X_val, y_val = X_shuffled[train_end:val_end], y_shuffled[train_end:val_end]
    X_test, y_test = X_shuffled[val_end:], y_shuffled[val_end:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# --- Execution Example ---
if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv("../data/student_performance.csv")
    X = df[['study_hours', 'attendance', 'previous_score', 'sleep']].values
    y = (df['exam_score'] >= 60).astype(int).values  # Classification target

    # 2. Split dataset (70% train, 15% validation, 15% test)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    print(f"Dataset Split Sizes:")
    print(f"Training Set:   {len(X_train)} samples")
    print(f"Validation Set: {len(X_val)} samples")
    print(f"Test Set:       {len(X_test)} samples\n")

    # 3. Simulate dummy predictions to showcase evaluation metrics
    # (Imagine a model predicting all 1s or random classes)
    y_pred_dummy = np.random.choice([0, 1], size=len(y_val))

    acc, prec, rec, f1 = compute_classification_metrics(y_val, y_pred_dummy)
    print("Evaluation Metrics on Validation Set:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")