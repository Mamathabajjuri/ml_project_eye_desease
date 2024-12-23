import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import os

# Define paths
csv_file = "C:\\Users\\pulim\\Downloads\\full_df.csv"  # Path to your CSV file
train_data_dir = "C:\\Users\\pulim\\OneDrive\\Desktop\\eye\\dataset\\train" # Path to your training images

# Parameters
batch_size = 32
img_height, img_width = 150, 150
num_classes = 8  # Adjust based on your number of categories

# Load CSV file
df = pd.read_csv(csv_file)

# Check columns
print("Columns in CSV file:", df.columns)

# Function to load images and labels from CSV
def load_data_from_csv(df, img_height, img_width):
    images = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = os.path.join(train_data_dir, row['filename'])
        print(f"Loading image: {img_path}")
        try:
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img)
            images.append(img_array)
            
            # Convert label list to a single label (assuming one-hot encoded format in 'target' column)
            label = row['target']
            if isinstance(label, str):
                label = eval(label)  # Convert string representation to list
            labels.append(label)
        except FileNotFoundError:
            print(f"File not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    print("Finished loading images and labels.")
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    return images, labels

# Load data
images, labels = load_data_from_csv(df, img_height, img_width)

# Limit to 100 photos per category
class_counts = pd.Series(np.argmax(labels, axis=1)).value_counts()
limit = 100
limited_data = []
limited_labels = []
for class_id, count in class_counts.items():
    class_indices = np.where(np.argmax(labels, axis=1) == class_id)[0]
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
model.fit(limited_data, limited_labels, epochs=10, batch_size=batch_size, validation_split=0.2)

# Save the model
model_save_path = "C:\\Users\\pulim\\OneDrive\\Desktop\\eye\\model\\full_model1.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
