import tensorflow as tf
import cv2
import numpy as np
#from keras.models import load_model 
from load_tf_data import class_names
from training import model


def start():
    #model=load_model('ocean.h5')
    ocean_picture=("./nasa_images/testing_images/image_jpeg(90).jpg")
    img = tf.keras.utils.load_img(ocean_picture, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image is most likely an ocean with  {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))


    img_1 = cv2.imread("./nasa_images/testing_images/image_jpeg(90).jpg")
    imgray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray_1, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(f"The Total Number of Contours in the Image = {str(len(contours))} ")
    #command len used to calculate the number of contours in the image
    print(contours[0])
    cv2.drawContours(img_1, contours, -1,(0,2550,0),3)
    cv2.drawContours(imgray_1, contours, -1,(0,255,0),3)
    cv2.imshow('Image', img_1)
    cv2.imshow('Image GRAY', imgray_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

