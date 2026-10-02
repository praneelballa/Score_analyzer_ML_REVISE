import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

if __name__ == "__main__":
    print("Building the exact same Neural Network using TensorFlow...\n")
    
    # 1. Create mock student data with 6 features for 100 students
    np.random.seed(42)
    m = 100
    X = np.random.randn(m, 6)
    y = np.random.randint(0, 2, size=(m, 1)) # Binary Pass/Fail target (0 or 1)

    # 2. Build the model using Keras Sequential API
    # Notice how closely this matches our architecture from Phase 5:
    # - 6 inputs -> Hidden 1 (25 neurons) -> Hidden 2 (15 neurons) -> Output (1 neuron)
    model = Sequential([
        Input(shape=(6,)),
        Dense(25, activation='sigmoid'),
        Dense(15, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    # 3. Compile the model
    # - optimizer='adam': Automatically handles learning rate and optimization
    # - loss='binary_crossentropy': The exact binary cost function we used earlier
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("TensorFlow Model Architecture Summary:")
    model.summary()

    # 4. Train the model
    print("\nTraining TensorFlow model...")
    history = model.fit(X, y, epochs=10, batch_size=8, verbose=1)

    # 5. Make a prediction on a sample student
    sample_student = np.array([[0.5, -1.2, 1.1, 0.2, 0.8, -0.5]])
    prediction = model.predict(sample_student)
    
    print(f"\nTensorFlow Pass Probability for sample student: {prediction[0][0]*100:.2f}%")