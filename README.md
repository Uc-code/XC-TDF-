**#CODE NOT ALLOWED TO BE USED WITHOUT THE CONSENT OF THE AUTHORS**
**#XC-TDF Implementation on Egde_IIoT datset**

# Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense




# Ensure the visualization settings from your initial setup
plt.rcParams['figure.figsize'] = (8, 5.28)
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = 600


# Step 2: Load your dataset
df1 = pd.read_csv('/content/Backdoor_attack.csv')
df2 = pd.read_csv('/content/DDoS_HTTP_Flood_attack.csv')
df3 = pd.read_csv('/content/MITM_attack.csv')
df4 = pd.read_csv('/content/Modbus.csv')
df5 = pd.read_csv('/content/Ransomware_attack.csv')
df6 = pd.read_csv('/content/OS_Fingerprinting_attack.csv')
df7 = pd.read_csv('/content/XSS_attack.csv')


# Concatenate the DataFrames
merged_df = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)
   # Display the merged DataFrame
merged_df.head()


# Get the unique class names from the 'Attack_type' column
class_names = merged_df['Attack_type'].unique()

# Print the class names
print("Class Names:")
for class_name in class_names:
    print(class_name)

# Drop the 'Attack_label' column
merged_df = merged_df.drop('Attack_label', axis=1)

# Drop the specified columns
columns_to_drop = ['frame.time', 'ip.src_host', 'ip.dst_host']
merged_df = merged_df.drop(columns_to_drop, axis=1)

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'Attack_type' column
merged_df['Attack_type'] = label_encoder.fit_transform(merged_df['Attack_type'])
merged_df

# Encode non-numeric columns
non_numeric_columns = merged_df.select_dtypes(exclude=['float64', 'int64']).columns
for column in non_numeric_columns:
    merged_df[column] = merged_df[column].astype(str)  # Convert to string
    merged_df[column] = label_encoder.fit_transform(merged_df[column])

# Compute the correlation matrix
correlation_matrix = merged_df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap Before Feature Selection')
plt.show()

# Select features and target variable
X = merged_df.drop(columns=['Attack_type'])
y = merged_df['Attack_type']


# Train a Random Forest classifier to determine feature importances
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X, y)

# Set the threshold to select the top 21 features
threshold = -np.sort(-rf_classifier.feature_importances_)[20]  # Get the importance of the 20th feature
feature_selector = SelectFromModel(rf_classifier, threshold=threshold)
feature_selector.fit(X, y)

# Get the names of selected features
selected_feature_names = X.columns[feature_selector.get_support(indices=True)]

# Create a heatmap to visualize the correlation between the selected features
selected_features_df = pd.DataFrame(feature_selector.transform(X), columns=selected_feature_names)
correlation_matrix = selected_features_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap after Features Selected")
plt.show()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train_selected, X_test_selected, y_train, y_test = train_test_split(selected_features_df, y, test_size=0.2, random_state=42)


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)



**#BASELINE DNN**

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=21, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# Define your optimizer
optimizer = Adam(learning_rate=0.0001)
# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Start recording the training time
start_time = time.time()

# Model training with the clean samples
history = ann.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_data=(X_test_scaled, y_test))

# Get predictions on the test data
start_time = time.time()
y_pred = ann.predict(X_test_scaled)
end_time = time.time()

# Convert predicted probabilities to classes
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_classes)
mcc = matthews_corrcoef(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')
prediction_time = end_time - start_time

# Print the metrics
print("Accuracy:", accuracy)
print("MCC:", mcc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Prediction Time:", prediction_time, "seconds")




**#GAUSSIAN NOISE-INJECTED DNN**

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=21, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# Define your optimizer
optimizer = Adam(learning_rate=0.0001)
# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Start recording the training time
start_time = time.time()

# Define parameters
epochs = 5
epsilon = 0.05  # Noise level 
iteration = 5  # Number of iterations

# Create copies of the original training and testing data
X_train_noisy = X_train_scaled.copy()
X_test_noisy = X_test_scaled.copy()

# Add Gaussian noise to the testing data for noisy predictions
noise_test = np.random.normal(0, epsilon, X_test_noisy.shape)
X_test_noisy += noise_test  # Noisy test data

# Initialize lists to store evaluation metrics
accuracy_list = []
recall_list = []
precision_list = []
f1_list = []
mcc_list = []
conf_matrix_list = []
train_accuracy_history = []
val_accuracy_history = []
train_loss_history = []
val_loss_history = []

# Train the model on noisy training data
for i in range(iteration):
    # Generate Gaussian noise for the training data
    noise_train = np.random.normal(0, epsilon, X_train_noisy.shape)
    X_train_noisy += noise_train  # Noisy training data

    # Train the model on noisy data
    history = ann.fit(X_train_noisy, y_train, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

    # Store training and validation metrics
    train_accuracy_history.extend(history.history['accuracy'])
    val_accuracy_history.extend(history.history['val_accuracy'])
    train_loss_history.extend(history.history['loss'])
    val_loss_history.extend(history.history['val_loss'])
    
# Evaluate on noisy test data
print("\nEvaluation on Noisy Test Data:")
start_time = time.time()
y_pred_noisy = ann.predict(X_test_noisy)
end_time = time.time()

y_pred_classes_noisy = np.argmax(y_pred_noisy, axis=1)
accuracy_noisy = accuracy_score(y_test, y_pred_classes_noisy)
mcc_noisy = matthews_corrcoef(y_test, y_pred_classes_noisy)
precision_noisy = precision_score(y_test, y_pred_classes_noisy, average='weighted')
recall_noisy = recall_score(y_test, y_pred_classes_noisy, average='weighted')
f1_noisy = f1_score(y_test, y_pred_classes_noisy, average='weighted')

print("Accuracy:", accuracy_noisy)
print("MCC:", mcc_noisy)
print("Precision:", precision_noisy)
print("Recall:", recall_noisy)
print("F1 Score:", f1_noisy)
print("Prediction Time:", end_time - start_time, "seconds")



**#DNN ON NOISE AND ADVERSRIAL SAMPLES BASED ON CIGM**

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=21, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# Define your optimizer
optimizer = Adam(learning_rate=0.0001)
# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Start recording the training time
start_time = time.time()

# Define parameters
epochs = 5
epsilon = 0.05 # Adversrial perturbation magnitutde
noise_level = 0.05  # Noise level 

# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to add noise to the samples
def add_noise(samples, noise_level=0.05):
    noise = np.random.normal(0, noise_level, samples.shape)
    noisy_samples = samples + noise
    return np.clip(noisy_samples, 0, 1)  # Clip to valid range

# Function to generate adversarial samples using the controlled iterative gradient method
def controlled_iterative_gradient_method(original_samples, epsilon, iterations):
    adversarial_samples = original_samples.copy()
    num_samples, num_features = original_samples.shape

    for _ in range(iterations):
        # Select a random feature to perturb
        feature_index = np.random.randint(num_features)

        # Perturb the selected feature using uniform random values
        perturbation = np.random.uniform(-epsilon, epsilon, size=num_samples)
        adversarial_samples[:, feature_index] += perturbation

        # Clip the perturbed samples to ensure they remain within valid input range
        adversarial_samples = np.clip(adversarial_samples, 0, 1)

        # Round the perturbed values to the nearest integer to preserve data type
        adversarial_samples = np.round(adversarial_samples)

    return adversarial_samples

# Create copies of the original training and testing data
X_train_noisy = add_noise(X_train_scaled.copy(), noise_level=noise_level)
X_test_noisy = add_noise(X_test_scaled.copy(), noise_level=noise_level)

# Generate adversarial samples using the controlled iterative gradient method
adversarial_train_samples = controlled_iterative_gradient_method(X_train_noisy, epsilon, epochs)
adversarial_test_samples = controlled_iterative_gradient_method(X_test_noisy, epsilon, epochs)

# Combine original, noisy, and adversarial samples for training
combined_train_samples = np.concatenate((X_train_noisy, adversarial_train_samples), axis=0)
combined_train_labels = np.concatenate((y_train, y_train), axis=0)  # Adjust labels accordingly

# Train the model on the combined samples
start_time_train = time.time()
history = ann.fit(combined_train_samples, combined_train_labels, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)
end_time_train = time.time()

# Store training history
train_accuracy_history = history.history['accuracy']
val_accuracy_history = history.history['val_accuracy']
train_loss_history = history.history['loss']
val_loss_history = history.history['val_loss']

# Store training time
train_time = end_time_train - start_time_train

# Store prediction time
start_time_predict = time.time()

# Combine noisy and adversarial test samples
combined_test_samples = np.concatenate((X_test_noisy, adversarial_test_samples), axis=0)
combined_test_labels = np.concatenate((y_test, y_test), axis=0)  # Adjust labels accordingly

# Evaluate the model on the combined test data
y_pred = ann.predict(combined_test_samples)
y_pred_classes = np.argmax(y_pred, axis=1)

end_time_predict = time.time()
prediction_time = end_time_predict - start_time_predict

# Calculate evaluation metrics for the test data
accuracy = accuracy_score(combined_test_labels, y_pred_classes)
recall = recall_score(combined_test_labels, y_pred_classes, average='weighted')
precision = precision_score(combined_test_labels, y_pred_classes, average='weighted')
f1 = f1_score(combined_test_labels, y_pred_classes, average='weighted')
mcc = matthews_corrcoef(combined_test_labels, y_pred_classes)
conf_matrix = confusion_matrix(combined_test_labels, y_pred_classes)

# Print evaluation metrics
print(f"Test Accuracy: {accuracy:.4f}, Test Recall: {recall:.4f}, Test Precision: {precision:.4f}, "
      f"Test F1 Score: {f1:.4f}, Test MCC: {mcc:.4f}")
print(f"Prediction Time: {prediction_time:.4f} seconds")
print(f"Training Time: {train_time:.4f} seconds")







**#ADVERSRIAL TRAINING ON DNN**

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=21, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# Define your optimizer
optimizer = Adam(learning_rate=0.0001)
# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Start recording the training time
start_time = time.time()

# Define parameters
epochs = 5
epsilon = 0.07 # Adversrial perturbation magnitutde
noise_level = 0.05  # Noise level 

# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to add noise to the samples
def add_noise(samples, noise_level=0.05):
    noise = np.random.normal(0, noise_level, samples.shape)
    noisy_samples = samples + noise
    return np.clip(noisy_samples, 0, 1)  # Clip to valid range

# Function to generate adversarial samples using the controlled iterative gradient method
def controlled_iterative_gradient_method(original_samples, epsilon, iterations):
    adversarial_samples = original_samples.copy()
    num_samples, num_features = original_samples.shape

    for _ in range(iterations):
        # Select a random feature to perturb
        feature_index = np.random.randint(num_features)

        # Perturb the selected feature using uniform random values
        perturbation = np.random.uniform(-epsilon, epsilon, size=num_samples)
        adversarial_samples[:, feature_index] += perturbation

        # Clip the perturbed samples to ensure they remain within valid input range
        adversarial_samples = np.clip(adversarial_samples, 0, 1)

        # Round the perturbed values to the nearest integer to preserve data type
        adversarial_samples = np.round(adversarial_samples)

    return adversarial_samples

# Create copies of the original training and testing data
X_train_noisy = add_noise(X_train_scaled.copy(), noise_level=noise_level)
X_test_noisy = add_noise(X_test_scaled.copy(), noise_level=noise_level)

# Generate adversarial samples using the controlled iterative gradient method
adversarial_train_samples = controlled_iterative_gradient_method(X_train_scaled, epsilon, epochs)  # Clean data for adversarial generation
adversarial_test_samples = controlled_iterative_gradient_method(X_test_scaled, epsilon, epochs)  # Clean data for adversarial generation

# Combine original (clean), noisy, and adversarial samples for training
combined_train_samples = np.concatenate((X_train_scaled, X_train_noisy, adversarial_train_samples), axis=0)
combined_train_labels = np.concatenate((y_train, y_train, y_train), axis=0)  # Adjust labels accordingly

# Train the model on the combined samples (clean, noisy, adversarial)
start_time_train = time.time()
history = ann.fit(combined_train_samples, combined_train_labels, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)
end_time_train = time.time()

# Store training history
train_accuracy_history = history.history['accuracy']
val_accuracy_history = history.history['val_accuracy']
train_loss_history = history.history['loss']
val_loss_history = history.history['val_loss']

# Store training time
train_time = end_time_train - start_time_train

# Store prediction time
start_time_predict = time.time()

# Combine noisy and adversarial test samples (also including clean test data)
combined_test_samples = np.concatenate((X_test_scaled, X_test_noisy, adversarial_test_samples), axis=0)
combined_test_labels = np.concatenate((y_test, y_test, y_test), axis=0)  # Adjust labels accordingly

# Evaluate the model on the combined test data
y_pred = ann.predict(combined_test_samples)
y_pred_classes = np.argmax(y_pred, axis=1)

end_time_predict = time.time()
prediction_time = end_time_predict - start_time_predict

# Calculate evaluation metrics for the test data
accuracy = accuracy_score(combined_test_labels, y_pred_classes)
recall = recall_score(combined_test_labels, y_pred_classes, average='weighted')
precision = precision_score(combined_test_labels, y_pred_classes, average='weighted')
f1 = f1_score(combined_test_labels, y_pred_classes, average='weighted')
mcc = matthews_corrcoef(combined_test_labels, y_pred_classes)
conf_matrix = confusion_matrix(combined_test_labels, y_pred_classes)

# Print evaluation metrics
print(f"Test Accuracy: {accuracy:.4f}, Test Recall: {recall:.4f}, Test Precision: {precision:.4f}, "
      f"Test F1 Score: {f1:.4f}, Test MCC: {mcc:.4f}")
print(f"Prediction Time: {prediction_time:.4f} seconds")
print(f"Training Time: {train_time:.4f} seconds")






**#TIME COMPARSION OF CIGM VERSUS FGSM/BIM**

# Define the model
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=21, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to generate adversarial examples using FGSM
def fgsm(model, features, epsilon=0.1):
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(features_tensor)
        predictions = model(features_tensor)
        loss = tf.keras.losses.binary_crossentropy(tf.ones_like(predictions), predictions)
    gradient = tape.gradient(loss, features_tensor)
    perturbed_features = features_tensor + epsilon * tf.sign(gradient)
    return perturbed_features.numpy()

# Function to generate adversarial examples using BIM
def bim(model, features, epsilon=0.1, iterations=5):
    perturbed_features = tf.identity(features)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_features)
            predictions = model(perturbed_features)
            loss = tf.keras.losses.binary_crossentropy(tf.ones_like(predictions), predictions)
        gradient = tape.gradient(loss, perturbed_features)
        perturbed_features = perturbed_features + epsilon * tf.sign(gradient)
        perturbed_features = tf.clip_by_value(perturbed_features, 0, 1)
    return perturbed_features.numpy()

# Function to generate adversarial examples using CIGM
def controlled_iterative_gradient_method(original_samples, epsilon=0.1, iterations=5):
    adversarial_samples = original_samples.copy()
    num_samples, num_features = original_samples.shape
    for _ in range(iterations):
        num_features_to_perturb = np.random.randint(1, num_features + 1)
        feature_indices_to_perturb = np.random.choice(num_features, num_features_to_perturb, replace=False)
        perturbation = np.random.uniform(-epsilon, epsilon, size=(num_samples, num_features_to_perturb))
        adversarial_samples[:, feature_indices_to_perturb] += perturbation
        adversarial_samples = np.clip(adversarial_samples, -1, 1)
        adversarial_samples = np.round(adversarial_samples)
    return adversarial_samples

# Original features for adversarial examples
original_features = np.random.randn(400000, 21)  # Example data (600000 samples, 21 features)

# Generating adversarial examples using FGSM
start_time_fgsm = time.time()
perturbed_fgsm = fgsm(ann, original_features)
end_time_fgsm = time.time()

# Generating adversarial examples using BIM
start_time_bim = time.time()
perturbed_bim = bim(ann, original_features)
end_time_bim = time.time()

# Generating adversarial examples using CIGM
start_time_cigm = time.time()
perturbed_cigm = controlled_iterative_gradient_method(original_features)
end_time_cigm = time.time()

# Print time comparisons for adversarial generation methods
print("Time for FGSM:", end_time_fgsm - start_time_fgsm)
print("Time for BIM:", end_time_bim - start_time_bim)
print("Time for CIGM:", end_time_cigm - start_time_cigm)

# Plotting the time comparison
methods = ['FGSM', 'BIM', 'CIGM']
times = [end_time_fgsm - start_time_fgsm, end_time_bim - start_time_bim, end_time_cigm - start_time_cigm]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, times, color=['blue', 'green', 'red'])
plt.xlabel('Adversarial Methods')
plt.ylabel('Time (seconds)')
plt.title('Time Comparison for Adversarial Transformation')

# Adding the time values as annotations on the bars
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{time_val:.2f}', ha='center', va='bottom')

plt.show()

# Print the model summary
ann.summary()



**#XAI**

**#GLOBAL EXPLANATION USING SHAP**
pip install shap

# Calculate the number of samples to take as 10% of the test set for quicker computation
num_samples = int(0.10 *combined_test_samples.shape[0])

# Select a sample of data to explain
sample_indices = np.random.choice(combined_test_samples.shape[0], num_samples, replace=False)
X_sample = combined_test_samples[sample_indices]

# Initialize the SHAP explainer with a smaller background dataset
background_data = shap.sample(X_train_scaled, 5) 
explainer = shap.KernelExplainer(model=ann.predict, data=background_data)

# Calculate SHAP values for the sample
shap_values = explainer.shap_values(X_sample)

# Global summary plot


**#LOCAL EXPLANATION USING SHAP**

import lime
import lime.lime_tabular
from sklearn.pipeline import make_pipeline

# Define your classes
class_labels = ['Backdoor', 'DDoS_HTTP', 'MITM', 'Normal', 'Ransomware', 'OS_Fingerprinting', 'XSS']

# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(combined_test_samples, mode="classification", feature_names=selected_feature_names, class_names=class_labels, discretize_continuous=False)

# Define a function to predict using the model
predict_fn = lambda x: ann.predict(x)

# Explain a single prediction
explanation = explainer.explain_instance(combined_test_samples[19], predict_fn, num_features=10, top_labels=len(class_labels), num_samples=1000)

# Show the explanation
explanation.show_in_notebook()



**#XC-TDF COMPUTATIONAL COMPLEXITY**


**Model Size** (Memory Footprint): The total size of the trained model file, which indicates how much memory is required to store the model on disk.

import joblib
import sys
import os
# Save the trained model to a file
model_filename = "dnn_model.h5"  # For Keras models, it's typically saved with .h5 extension
ann.save(model_filename)

# Calculate the size of the saved model file
model_size_bytes = os.path.getsize(model_filename)

# Convert to kilobytes (KB) or megabytes (MB) for better readability
model_size_kb = model_size_bytes / 1024
model_size_mb = model_size_kb / 1024

print(f"Size of the trained DNN model: {model_size_bytes} bytes, {model_size_kb:.2f} KB, {model_size_mb:.4f} MB")

# Clean up (optional)
os.remove(model_filename)

# Memory usage of the entire DNN model (in memory)
model_memory = sys.getsizeof(ann)

# If your DNN model is very large and consists of many layers and parameters,
# you can sum the memory usage of each layer's parameters:
for layer in ann.layers:
    model_memory += sys.getsizeof(layer.get_weights())

print(f"Memory footprint of the DNN model: {model_memory / (1024 ** 2):.4f} MB")



**Training Time**:The time taken to train the model. This shows how much computational resource is required to train the model.

import time
# Measures time taken to predict on the entire test set
start_time = time.time()
y_pred = ann.predict(combined_test_samples)
end_time = time.time()

# Calculates the time taken and throughput
prediction_time = end_time - start_time
number_of_predictions = len(combined_test_samples)

throughput = number_of_predictions / prediction_time

print(f"Time taken for prediction: {prediction_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} predictions/second")


**Prediction Time** (Inference Time):The time taken to make predictions for a given test set, which reflects how fast the model can generate predictions.
start_time = time.time()
predictions = ann.predict(combined_test_samples)
end_time = time.time()

prediction_time = end_time - start_time
print(f"Prediction time: {prediction_time:.4f} seconds")


**Throughput** (Predictions per Second):This metric indicates how many predictions the model can make per second. A higher throughput indicates better computational efficiency.
num_predictions = len(combined_test_samples)
throughput = num_predictions / prediction_time
print(f"Throughput: {throughput:.2f} predictions/second")
