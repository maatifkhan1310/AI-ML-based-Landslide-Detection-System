import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img

def display_images_with_predictions(images, labels, predictions, class_names):
    """
    Display images with true labels and predicted labels.
    
    Parameters:
        images (numpy array): Array of images.
        labels (numpy array): Array of true labels.
        predictions (numpy array): Array of predicted labels.
        class_names (list): List of class names corresponding to the labels.
    """
    plt.figure(figsize=(15, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_image_and_prediction(image_path, model, class_names):
    """
    Plot a single image with its predicted label.
    
    Parameters:
        image_path (str): Path to the image file.
        model (keras.Model): Trained Keras model.
        class_names (list): List of class names corresponding to the labels.
    """
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_label = class_names[int(prediction[0] > 0.5)]
    
    plt.imshow(load_img(image_path))
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()
