import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


img_width = 640
img_height = 480

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_annot = pd.read_csv("train_annot.csv")
test_annot = pd.read_csv("test_annot.csv")

train_generator = train_datagen.flow_from_dataframe(dataframe=train_annot, directory="train_data", class_mode="raw",
                                                    x_col="filename", y_col=["x", "y", "theta"],
                                                    target_size=(img_height, img_width), batch_size=1, featurewise_center=True)

test_generator = train_datagen.flow_from_dataframe(dataframe=test_annot, directory="test_data", class_mode="raw",
                                                   x_col="filename", y_col=["x", "y", "theta"],
                                                   target_size=(img_height, img_width), batch_size=1, featurewise_center=True)

model_resnet = tf.keras.applications.resnet50.ResNet50(input_shape=(img_height, img_width, 3), include_top=False,
                                                      weights='imagenet')

model_resnet.trainable = False


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
inbetween_layer = tf.keras.layers.Dense(1024, activation='relu')
prediction_layer = tf.keras.layers.Dense(3, activation='linear')

model = tf.keras.Sequential([
  model_resnet, global_average_layer, inbetween_layer, prediction_layer])

model.compile(loss='mae', optimizer='adam')
# model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

history = model.fit(train_generator, validation_data=test_generator, steps_per_epoch=100, epochs=300,
                    validation_steps=10, verbose=1)
model.save('resnet-newdata-tl.h5')

model.evaluate(test_generator)

epochs_range = range(300)

loss = history.history['loss']
val_loss = history.history['val_loss']
print("Val_Loss:", val_loss)

plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('resnet-newdata-tl-loss.png')
