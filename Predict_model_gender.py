import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import splitfolders
import tensorflow_hub as hub
import cv2
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

modelo =tf.keras.models.load_model('save_dir_logs/model_gender')
classes=['men','women']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
#.----------------Cam predict-------------------.#
cam = cv2.VideoCapture(0)
i=0

while(i<1000000):
    ret,frame = cam.read()
    print(frame.shape[0],frame.shape[1])
    image=tf.image.resize(frame,size=(180,180))
    image = image/255.
    print(image.shape[0],image.shape[1])
    prediccion = tf.squeeze(modelo.predict(tf.expand_dims(image,axis=0)))
    #score = tf.reduce_max(prediccion)*100
    prediccion_class = classes[int(prediccion)]
    texto = f"Clase: {prediccion_class}"
    print(f"PREDICION: {modelo.predict(tf.expand_dims(image,axis=0))}")
    if prediccion < 0.5 or prediccion >0.8:
        cv2.putText(frame,texto,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,200),2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #bodys = body_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #for (x, y, w, h) in bodys:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    cv2.imshow("Gender_Classificator",frame)
    cv2.waitKey(10)
    i+=1
cv2.destroyAllWindows()