import tensorflow as tf
import numpy as np
import os
import time
from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback
from tf_explain.callbacks.grad_cam import GradCAMCallback
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print(f"There are {len(train_images)} with shape {train_images.shape},\n with type{type(train_images)} and dtype {train_images.dtype}")

train_images = train_images/255.0
test_images = test_images/255.0

print(np.max(train_images[0]),np.min(train_images[0]))
img = tf.constant(test_images)
label = tf.constant(test_labels)

#-----------------------------Create dataset with tf.data.Dataset-----------------------#
# Turn data into efficient running data
train = tf.data.Dataset.from_tensor_slices((train_images,tf.cast(train_labels,dtype=tf.float32)))
trainDS = (train.shuffle(32 * 100).batch(32).prefetch(tf.data.AUTOTUNE))
test = tf.data.Dataset.from_tensor_slices((test_images,tf.cast(test_labels,dtype=tf.float32)))
testDS = (test.shuffle(32 * 100).batch(32).prefetch(tf.data.AUTOTUNE))

classes = sorted(set(train_labels))
print(classes)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

before = time.time()
model_mnist = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # input layer (we had to reshape 28x28 to 784)
  tf.keras.layers.Dense(512, activation="tanh"),
  tf.keras.layers.Dense(512, activation="tanh"),
  tf.keras.layers.Dense(256, activation="tanh"),
  tf.keras.layers.Dense(10, activation="softmax") # output shape is 10, activation is softmax
])
model_mnist.summary()
model_mnist.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=["accuracy"])
model_mnist.fit(trainDS,
                epochs=100,
                validation_data=testDS,
                callbacks=([tf.keras.callbacks.ModelCheckpoint('save_dir_logs/model_fashion',save_best_only=True),
                           tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),
                           tf.keras.callbacks.TensorBoard('experiments_metrics/model_fashion'),
                           ActivationsVisualizationCallback(validation_data=(img,label),layers_name=["activation_1"],output_dir='fashion_ouput_1'),
                           GradCAMCallback(validation_data=(img,label),layer_name="activation_2",class_index=0,output_dir='fashion_ouput_2')]))
print(time.time()-before,"segundos")
