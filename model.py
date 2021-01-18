import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 16
img_height = 180
img_width = 180


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, input_shape=(img_width, img_height, 3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Dropout(0.2),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Dropout(0.3),
  layers.Flatten(),
  layers.Dense(64, activation="relu"),
  layers.Dropout(0.5),
  layers.Dense(5, activation="softmax")
])

model.compile(
  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-06),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=4
)