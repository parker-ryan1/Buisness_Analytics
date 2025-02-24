import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import linear_sum_assignment

# Define paths
dataset_dir = "dataset"  # Path to the dataset folder
img_height, img_width = 128, 128  # Image dimensions
batch_size = 32

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load data
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Number of couples
num_classes = len(train_generator.class_indices)
print(f"Number of couples: {num_classes}")

# Build the Siamese Network
def build_embedding_model():
    base_model = applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False  # Freeze the base model initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
    ])
    return model

embedding_model = build_embedding_model()

# Siamese Network
input_1 = layers.Input(shape=(img_height, img_width, 3))
input_2 = layers.Input(shape=(img_height, img_width, 3))

embedding_1 = embedding_model(input_1)
embedding_2 = embedding_model(input_2)

# Euclidean distance between embeddings
distance = layers.Lambda(lambda x: tf.math.reduce_sum(tf.math.square(x[0] - x[1]), axis=1))([embedding_1, embedding_2])

siamese_network = models.Model(inputs=[input_1, input_2], outputs=distance)

# Compile the Siamese Network
siamese_network.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="binary_crossentropy")

# Generate pairs for training
def generate_pairs(images, labels, num_pairs=1000):
    pairs = []
    pair_labels = []
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(len(images), 2)
        if labels[idx1] == labels[idx2]:
            pairs.append([images[idx1], images[idx2]])
            pair_labels.append(1)  # Positive pair
        else:
            pairs.append([images[idx1], images[idx2]])
            pair_labels.append(0)  # Negative pair
    return np.array(pairs), np.array(pair_labels)

# Load all images and labels
def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(int(label.split("_")[1]))  # Extract couple number from folder name
    return np.array(images), np.array(labels)

# Load dataset
train_images, train_labels = load_images_and_labels(dataset_dir)

# Generate pairs for training
pairs, pair_labels = generate_pairs(train_images, train_labels, num_pairs=5000)

# Train the Siamese Network
siamese_network.fit(pairs, pair_labels, batch_size=32, epochs=10)

# Fine-tune the base model
embedding_model.trainable = True
for layer in embedding_model.layers[:-10]:  # Unfreeze the last 10 layers
    layer.trainable = False
siamese_network.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss="binary_crossentropy")
siamese_network.fit(pairs, pair_labels, batch_size=32, epochs=5)

# Save the embedding model
embedding_model.save("embedding_model.h5")

# Clustering for New Data
def cluster_couples(new_images):
    embeddings = embedding_model.predict(new_images)
    kmeans = KMeans(n_clusters=num_classes)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

# Evaluate Clustering
def evaluate_clustering(true_labels, predicted_labels):
    cost_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    accuracy = cost_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy

# Example: Load new images and evaluate clustering
new_images, new_labels = load_images_and_labels("new_dataset")  # Replace with path to new dataset
predicted_labels = cluster_couples(new_images)
accuracy = evaluate_clustering(new_labels, predicted_labels)
print(f"Clustering Accuracy: {accuracy}")

# Classification Report
print(classification_report(new_labels, predicted_labels))

# Confusion Matrix
print(confusion_matrix(new_labels, predicted_labels))

# Grad-CAM for Explainability
def grad_cam(model, image, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(output * weights, axis=-1)
    cam = tf.maximum(cam, 0)  # ReLU
    cam = cam / tf.reduce_max(cam)  # Normalize
    return cam.numpy()

# Example: Visualize Grad-CAM for an image
image = new_images[0][np.newaxis, ...]
cam = grad_cam(embedding_model, image, "global_average_pooling2d")
plt.imshow(cam, cmap="jet")
plt.show()
