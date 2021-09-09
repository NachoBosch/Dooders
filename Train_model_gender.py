import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import splitfolders
import tensorflow_hub as hub
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


dir_gender = 'data_gender/'
for dir,dirnames,files in os.walk(dir_gender):
    print(f"There are {len(files)} in {dirnames} of {dir}")
classes = [f for f in os.listdir(dir_gender)]
print(classes)
#splitfolders.ratio(dir_gender, output="gender_dataset", seed=1337, ratio=(.8, .1, .1), group_prefix=None)

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,horizontal_flip=True)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_batch = train_gen.flow_from_directory('gender_dataset/train/',
                                            target_size=(180,180),
                                            batch_size=32,
                                            class_mode='binary')
test_batch = test_gen.flow_from_directory('gender_dataset/test/',
                                          batch_size=32,
                                          target_size=(180,180),
                                          class_mode='binary')
val_batch = val_gen.flow_from_directory('gender_dataset/val',
                                        batch_size=32,
                                        target_size=(180,180),
                                        class_mode='binary')
images,labels = train_batch[0]
print(images[0].shape,labels[0])
plt.figure(figsize=(10,7))
plt.imshow(images[0])
plt.show()
base = hub.KerasLayer("https://tfhub.dev/google/efficientnet/b0/feature-vector/1",trainable=False,input_shape=(180,180,3))

model_gender = tf.keras.Sequential([
tf.keras.layers.Input(shape=(180,180,3)),
base,
tf.keras.layers.Dense(1,activation='sigmoid')])
model_gender.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
model_gender.summary()
model_gender.fit(train_batch,
                 epochs=5,
                 validation_data=val_batch,
                 callbacks=[tf.keras.callbacks.ModelCheckpoint('save_dir_logs/model_gender',save_best_only=True),
                            tf.keras.callbacks.TensorBoard('experiments_metrics/model_gender')])