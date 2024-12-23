import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np

# Define paths
csv_file = "C:\\Users\\S.JHANSY\\OneDrive\\Desktop\\EYE2\\EYE2\\full_df.csv"  # Path to your CSV file
train_data_dir = "C:\\Users\\S.JHANSY\\OneDrive\\Desktop\\EYE2\\EYE2\\dataset\\train" # Path to your training images

# Parameters
batch_size = 32
img_height, img_width = 150, 150
num_classes = 8  # Number of categories in your labels

# Load CSV file
df = pd.read_csv(csv_file)

# Print column names to verify
print("Columns in CSV file:", df.columns)

# Function to load images and labels from CSV
def load_data_from_csv(df, img_height, img_width):
    images = []
    labels = []
    
    # Define class names and create a mapping
    class_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    class_to_index = {name: index for index, name in enumerate(class_names)}
    
    for _, row in df.iterrows():
        img_path = f"{train_data_dir}/{row['filename']}"
        try:
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img)
            images.append(img_array)
            # Convert label list to a single label (assuming only one label per image)
            label = row['labels'].strip('[]').replace("'", "").split(", ")[0]
            labels.append(class_to_index.get(label, -1))  # Convert to integer label
        except FileNotFoundError:
            print(f"File not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    return images, labels

# Load data
images, labels = load_data_from_csv(df, img_height, img_width)

# Limit to 100 photos per category
class_counts = pd.Series(labels.argmax(axis=1)).value_counts()
limit = 100
limited_data = []
limited_labels = []
for class_id, count in class_counts.items():
    class_indices = np.where(labels.argmax(axis=1) == class_id)[0]
    selected_indices = class_indices[:limit]
    limited_data.append(images[selected_indices])
    limited_labels.append(labels[selected_indices])

limited_data = np.concatenate(limited_data)
limited_labels = np.concatenate(limited_labels)

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(limited_data, limited_labels, epochs=50, batch_size=batch_size, validation_split=0.2)

# Save the model
model_save_path = "C:\\Users\\S.JHANSY\\OneDrive\\Desktop\\EYE2\\EYE2\\model1\\full_model1.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
