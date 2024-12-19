import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150

def preprocess_image(image_path):
    """
    Preprocess a single image for model input.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        img_array (numpy array): Preprocessed image as a numpy array.
    """
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
