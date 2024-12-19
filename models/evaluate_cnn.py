import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
test_dir = 'dataset/test'
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Data generator for the test set
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Loss: {loss:.2f}")
