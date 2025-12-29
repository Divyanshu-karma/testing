import numpy
import cv2
import tensorflow as tf
import keras
from fer import FER
from mtcnn.mtcnn import MTCNN

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("FER:", FER.__version__)
print("MTCNN loaded successfully")
