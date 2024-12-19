import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150

def load_dataset(dataset_dir, class_labels):
    """
    Load the dataset and return images and labels as numpy arrays.
    
    Parameters:
        dataset_dir (str): Path to the dataset directory.
        class_labels (dict): Dictionary mapping class names to labels (e.g., {"landslide": 1, "no_landslide": 0}).
    
    Returns:
        X (numpy array): Array of images.
        y (numpy array): Array of labels.
    """
    images = []
    labels = []
    
    for label, class_name in class_labels.items():
        class_dir = os.path.join(dataset_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0  # Normalize the image
            images.append(img_array)
            labels.append(class_name)
    
    return np.array(images), np.array(labels)
