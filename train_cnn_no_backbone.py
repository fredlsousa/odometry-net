import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
import pandas as pd


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_annot = pd.read_csv("train_annot.csv")
test_annot = pd.read_csv("test_annot.csv")

train_generator = train_datagen.flow_from_dataframe(dataframe=train_annot, directory="train_data",
													class_mode="multi_output", x_col="filename", y_col=["x","y","theta"],
													target_size=(300, 200), batch_size=30)

test_generator = train_datagen.flow_from_dataframe(dataframe=test_annot, directory="test_data",
													class_mode="multi_output", x_col="filename", y_col=["x","y","theta"],
													target_size=(300, 200), batch_size=30)


model = tf.keras.models.Sequential([
	layers.experimental.preprocessing.Rescaling(1./255., input_shape=(300, 200, 3)),
	tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Flatten(),		# Daqui pra cima estão as caracteristicas da imagem
	tf.keras.layers.Dense(1024, activation='relu'),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(3, activation='linear')
	])

model.compile(loss='mae', optimizer='adam')


history = model.fit(train_generator, validation_data=test_generator, steps_per_epoch=100, epochs=300,
					validation_steps=10, verbose=1)

model.save('cnn_model.h5')

model.evaluate(test_generator)

epochs_range = range(300)

loss = history.history['loss']
val_loss = history.history['val_loss']
print("Val_Loss:", val_loss)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
