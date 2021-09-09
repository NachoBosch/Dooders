import tensorflow as tf
import pixellib
from pixellib.instance import instance_segmentation
import cv2
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

segmentation_model = instance_segmentation()
#segmentation_model.load_model('mask_rcnn_coco.h5')
segmentation_model.load_model('mask_rcnn_hair_0200.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    res = segmentation_model.segmentFrame(frame,show_bboxes=True)[1]

    cv2.imshow('Instance_segmentation',res)
    if cv2.waitKey(10) & 0xFF ==ord('q'):
        break
cap.release()
