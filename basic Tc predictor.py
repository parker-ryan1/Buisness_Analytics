# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
file_path = "/content/featurized[1].csv"  # Update this if needed
df = pd.read_csv(file_path)

# Clean all columns
for col in df.columns:
    if df[col].dtype == 'object':  # Check if the column contains strings
        df[col] = df[col].str.replace('.', '', regex=False)  # Remove periods
        df[col] = df[col].str.replace(',', '.', regex=False)  # Replace commas with periods
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN

# Handle missing values
df = df.fillna(df.mean())  # Fill missing values with the mean

# Verify the data
print(df.dtypes)  # Ensure all columns are numeric
print(df.isnull().sum())  # Check for remaining missing values

# Feature selection (Ensure 'critical_temp' is correct)
X = df.drop(columns=['Tc'])  # Features
y = df['Tc'].values  # Target variable (transition temperature)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)  # Transform test data using the same scaler

# Reshape data for CNN (Conv1D requires 3D input: (samples, time steps, features))
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build CNN Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.5),

    Conv1D(filters=32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Conv1D(filters=16, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer (predicting Tc)
])

# Compile the model
model.compile(optimizer='Adagrad',
              loss = 'mae',
              metrics = ['mae']
              )

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=25,  # Increase epochs
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Make predictions
predictions = model.predict(X_test).flatten()  # Flatten to 1D

# Find top 5 materials with the highest predicted Tc
top_indices = np.argsort(predictions)[-5:][::-1]  # Get indices of highest Tc values

# Print top materials
print("\nTop 5 Predicted High Tc Materials:")
for i in top_indices:
    print(f"Material Formuola: {i}, Predicted Tc: {predictions[i]:.2f}, Actual Tc: {y_test[i]:.2f}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
