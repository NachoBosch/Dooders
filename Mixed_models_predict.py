import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.fashion_mnist.load_data()
model_fashion = tf.keras.models.load_model('save_dir_logs/model_fashion_2')
model_gender = tf.keras.models.load_model('save_dir_logs/model_gender')

class_gender = ['men','women']
class_fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#.----------------Cam predict-------------------.#
cam = cv2.VideoCapture(0)
i=0
while(i<1000000):
    ret,frame = cam.read()
    print(frame.shape[0],frame.shape[1])
    image_gender=tf.image.resize(frame,size=(180,180))
    image_gender = image_gender/255.
    print(image_gender.shape[0],image_gender.shape[1])
    prediccion_gender = tf.squeeze(tf.round(model_gender.predict(tf.expand_dims(image_gender,axis=0))))
    prediccion_class_gender = class_gender[int(prediccion_gender)]


    image_fashion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(type(image_fashion),image_fashion.shape)
    image_fashion = cv2.resize(image_fashion,(28,28),interpolation=cv2.INTER_AREA)
    image_fashion = tf.expand_dims(image_fashion/255.0,axis=0)
    print(type(image_fashion), image_fashion.shape)
    prediccion_fashion = tf.argmax(tf.squeeze(tf.nn.softmax(model_fashion.predict(image_fashion)))).numpy()
    prediccion_class_fashion = class_fashion[prediccion_fashion]


    texto = f"Gender: {prediccion_class_gender}, fashion: {prediccion_class_fashion}"
    print(f"PREDICION Gender: {model_gender.predict(tf.expand_dims(image_gender,axis=0))}")
    print(f"PREDICION Fashion: {prediccion_class_fashion}")
    if prediccion_gender < 0.5 or prediccion_gender >0.8:
        cv2.putText(frame,texto,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,200),2)
    cv2.imshow("Gender_Classificator",frame)
    cv2.waitKey(10)
    i+=1
cv2.destroyAllWindows()