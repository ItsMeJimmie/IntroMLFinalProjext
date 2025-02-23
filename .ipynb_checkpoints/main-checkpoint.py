import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Path to folder containing the images
image_folder = r"C:\Users\coxji\PycharmProjects\pythonProject\Datasets"

# Define RGB-to-label mapping for segmentation map
terrain_classes = {
    (17, 141, 215): 0,  # Water
    (225, 227, 155): 1,  # Grassland
    (127, 173, 123): 2,  # Forest
    (185, 122, 87): 3,  # Hills
    (230, 200, 181): 4,  # Desert
    (150, 150, 150): 5,  # Mountain
    (193, 190, 175): 6  # Tundra
}


# Helper function to load and preprocess a single image set
def load_and_preprocess(base_name, image_size=(64, 64)):
    # Build file paths
    terrain_path = os.path.join(image_folder, f'{base_name}_t.png')
    height_path = os.path.join(image_folder, f'{base_name}_h.png')
    segmentation_path = os.path.join(image_folder, f'{base_name}_i2.png')

    # Load images
    terrain_image = imread(terrain_path)
    height_image = imread(height_path)
    segmentation_image = imread(segmentation_path)

    # Resize images
    terrain_image = resize(terrain_image, image_size, anti_aliasing=True, preserve_range=True).astype('uint8')
    height_image = resize(height_image, image_size, anti_aliasing=True, preserve_range=True).astype('uint16')
    segmentation_image = resize(segmentation_image, image_size, anti_aliasing=True, preserve_range=True).astype('uint8')

    # Convert segmentation map to labels
    labels = np.apply_along_axis(
        lambda rgb: terrain_classes.get(tuple(rgb), -1), 2, segmentation_image
    ).flatten()

    # Filter out invalid labels (-1)
    valid_idx = labels != -1

    # Flatten and filter features
    terrain_flat = terrain_image.reshape(-1, terrain_image.shape[-1])[valid_idx]
    height_flat = height_image.flatten()[valid_idx]
    features = np.hstack([terrain_flat, height_flat.reshape(-1, 1)])

    return features, labels[valid_idx]


def process_in_batches(batch_start, batch_end, image_size=(64, 64)):
    batch_features = []
    batch_labels = []
    for i in range(batch_start, batch_end + 1):
        base_name = f"{str(i).zfill(4)}"
        try:
            # Load and preprocess
            features, labels = load_and_preprocess(base_name, image_size=image_size)
            batch_features.append(features)
            batch_labels.append(labels)
        except FileNotFoundError:
            print(f"Image set {base_name} not found, skipping.")
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
    # Combine batch into numpy arrays
    return np.vstack(batch_features), np.hstack(batch_labels)


# Process images in manageable batches
batch_size = 500  # Number of images per batch
total_images = 1000  # Total number of images
X = []
y = []

for batch_start in range(1, total_images + 1, batch_size):
    batch_end = min(batch_start + batch_size - 1, total_images)
    print(f"Processing batch: {batch_start} to {batch_end}")
    batch_X, batch_y = process_in_batches(batch_start, batch_end)
    X.append(batch_X)
    y.append(batch_y)

# Convert to single numpy arrays
X = np.vstack(X)
y = np.hstack(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training SVM Model")
svm_model = SVC(kernel='rbf', C=1.0, verbose=True, max_iter=1000)
svm_model.fit(X_train, y_train)

# Step 5: Evaluate model
print("Preparing to evaluate")
y_pred = svm_model.predict(X_test)

print("Classification")
print(classification_report(y_test, y_pred, target_names=[
    "Water", "Grassland", "Forest", "Hills", "Desert", "Mountain", "Tundra"
]))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

matrix = confusion_matrix(y_test, y_pred)
class_accuracies = matrix.diagonal()/matrix.sum(axis=1)

print("Per-Class Accuracy:")
for idx, class_name in enumerate(["Water", "Grassland", "Forest", "Hills", "Desert", "Mountain", "Tundra"]):
    print(f"{class_name}: {class_accuracies[idx]:.2f}")



# Function to decode label map to RGB for visualization
def decode_segmentation_map(label_map, terrain_classes):
    # Reverse the mapping dictionary
    reverse_mapping = {v: k for k, v in terrain_classes.items()}
    height, width = label_map.shape
    rgb_map = np.zeros((height, width, 3), dtype=np.uint8)

    for label, rgb in reverse_mapping.items():
        rgb_map[label_map == label] = rgb

    return rgb_map

# Show ground truth and predicted segmentation
def visualize_segmentation(X_test, y_test, y_pred, index, terrain_classes, image_size=(64, 64)):
    # Reshape labels to the original image shape
    ground_truth = y_test[index * image_size[0] * image_size[1]:(index + 1) * image_size[0] * image_size[1]].reshape(image_size)
    prediction = y_pred[index * image_size[0] * image_size[1]:(index + 1) * image_size[0] * image_size[1]].reshape(image_size)

    # Decode to RGB maps for visualization
    ground_truth_rgb = decode_segmentation_map(ground_truth, terrain_classes)
    prediction_rgb = decode_segmentation_map(prediction, terrain_classes)

    # Display images
    plt.figure(figsize=(10, 5))

    # Plot ground truth
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(ground_truth_rgb)
    plt.axis('off')

    # Plot prediction
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(prediction_rgb)
    plt.axis('off')

    plt.show()

# Call visualization function for the first test image
print("Visualizing ground truth and predicted segmentation for the first test image.")
visualize_segmentation(X_test, y_test, y_pred, index=0, terrain_classes=terrain_classes, image_size=(64, 64))