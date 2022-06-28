import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing import image as image_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.applications import VGG16
model = VGG16(weights="imagenet")
model.summary()


def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)


show_image("data/doggy_door_images/happy_dog.jpg")


def load_and_process_image(image_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)

    # Load in the image with a target size of 224, 224
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    # Convert the image from a PIL format to a numpy array
    image = tf.keras.utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1, 224, 224, 3)
    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)
    # Print image's shape after processing
    print('Processed image shape: ', image.shape)
    return image


processed_image = load_and_process_image(
    "data/doggy_door_images/brown_bear.jpg")


def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))


def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))


readable_prediction("data/doggy_door_images/happy_dog.jpg")
readable_prediction("data/doggy_door_images/brown_bear.jpg")
readable_prediction("data/doggy_door_images/sleepy_cat.jpg")


def doggy_door(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    preds = model.predict(image)
    if 151 <= np.argmax(preds) <= 268:
        print("Doggy come on in!")
    elif 281 <= np.argmax(preds) <= 285:
        print("Kitty stay inside!")
    else:
        print("You're not a dog! Stay outside!")


doggy_door("data/doggy_door_images/brown_bear.jpg")
doggy_door("data/doggy_door_images/happy_dog.jpg")
doggy_door("data/doggy_door_images/sleepy_cat.jpg")
doggy_door("data/Train/download.jpg")
doggy_door("data/Train/download (1).jpg")
doggy_door("data/Train/download (2).jpg")
