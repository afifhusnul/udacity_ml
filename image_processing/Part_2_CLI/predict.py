import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as tfHub
import argparse
import numpy as np
import json
from PIL import Image


parser = argparse.ArgumentParser(description='Read Arguments from Terminal')
parser.add_argument('imagePath', help='image path', default='')
parser.add_argument('modelPath', help='model path', default='')
parser.add_argument('--top_k', help='top K value', default='5')
parser.add_argument('--category_names', help='get class label json file', default = None)
args = parser.parse_args()

IMAGE_PATH = args.imagePath
MODEL_PATH = args.modelPath
TOP_K = args.top_k
CATEGORY_NAMES = args.category_names

def process_image(imagePath):
    #define image 
    im = Image.open(imagePath)
    image = np.asarray(im)
        
    #set size and normalize
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()    
    
    #set dimension
    image = np.expand_dims(image, axis = 0)
    return image

def load_model(modelPath):
    imageModel = keras.models.load_model(modelPath, custom_objects={'KerasLayer':tfHub.KerasLayer}, compile=False)
    return imageModel

def predict(imagePath, imageModel, top_k):
    #get image and process it
    image = process_image(imagePath)

    #predict image
    top_k = int(top_k)
    prediction = imageModel.predict(image)
    probs, classes = tf.math.top_k(prediction, k=top_k)
    probs = probs.numpy()[0]
    classes = [str(num + 1) for num in classes.numpy()[0]]
    return probs, classes


def main(imagePath, modelPath, top_k, category_names):

    my_model = load_model(modelPath)
    probs, classes = predict(imagePath, my_model, top_k)

    if category_names != None:
        label_map = category_names
        with open(label_map, 'r') as f:
             class_names = json.load(f)
        classes = [class_names[label] for label in classes]

    return probs, classes

if __name__ == "__main__":
    probs, classes = main(IMAGE_PATH, MODEL_PATH, TOP_K, CATEGORY_NAMES)
    print(probs)
    print(classes)