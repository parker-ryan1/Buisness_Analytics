import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

# Step 1: Generate a simple dataset
def generate_dataset(num_samples):
    # Create a binary classification problem
    x = np.random.uniform(0, 2 * np.pi, num_samples)
    y = np.sin(x) > 0  # Classify based on the sign of sin(x)
    return x, y

# Generate 1000 samples
x_train, y_train = generate_dataset(1000)
x_test, y_test = generate_dataset(200)

# Step 2: Encode classical data into quantum circuits
def encode_data(x, qubit):
    """Encode classical data into a quantum circuit."""
    circuit = cirq.Circuit()
    circuit.append(cirq.rx(x)(qubit))
    circuit.append(cirq.ry(x)(qubit))  # Adding another rotation gate for better encoding
    return circuit

# Define a qubit
qubit = cirq.GridQubit(0, 0)

# Encode the training and test data
x_train_circuits = [encode_data(x, qubit) for x in x_train]
x_test_circuits = [encode_data(x, qubit) for x in x_test]

# Convert to TensorFlow Quantum tensors
x_train_tfq = tfq.convert_to_tensor(x_train_circuits)
x_test_tfq = tfq.convert_to_tensor(x_test_circuits)

# Step 3: Build a more complex quantum model
def create_quantum_model():
    """Create a more complex quantum model with multiple parameterized gates."""
    bit = cirq.GridQubit(0, 0)
    model_circuit = cirq.Circuit()
    theta = sympy.Symbol('theta')
    phi = sympy.Symbol('phi')
    model_circuit.append(cirq.ry(theta)(bit))
    model_circuit.append(cirq.rz(phi)(bit))  # Adding another parameterized gate
    return model_circuit

# Create the quantum model
model_circuit = create_quantum_model()

# Step 4: Define a more expressive observable (e.g., combination of Pauli-Z and Pauli-X)
observable = 0.5 * cirq.Z(qubit) + 0.5 * cirq.X(qubit)

# Step 5: Build a Keras model with TFQ
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string),
    tfq.layers.PQC(model_circuit, observable),
])

# Step 6: Compile and train the model with adjusted hyperparameters
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Lower learning rate
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model with more epochs
history = model.fit(x_train_tfq, y_train,
                    epochs=20,  # Increased number of epochs
                    batch_size=64,  # Adjusted batch size
                    validation_data=(x_test_tfq, y_test))

# Step 7: Evaluate the model
results = model.evaluate(x_test_tfq, y_test)
print("Test accuracy:", results[1])
