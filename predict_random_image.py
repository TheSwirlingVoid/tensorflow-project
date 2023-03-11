import tensorflow as tf
from tensorflow import keras, zeros, data
from keras import models, utils
import cv2
import pickle

def main():
    
    model = models.load_model("checkpoints/450classes/checkpoints_145.000000")
    user_input = input("Enter a file path to classify: ")

    with open("checkpoints/450classes/class_names.data", "rb") as f:
        class_names = pickle.load(f)

    image = cv2.imread(user_input)
    resized = cv2.resize(image, (224,224))
    img_array = utils.img_to_array(resized)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))

    print(class_names[prediction])

main()
