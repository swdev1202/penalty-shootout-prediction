from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
import numpy as np
import pandas as pd

# TRAIN? 1, TEST? 0
TRAIN = 1

training_data_dir = "data/training"
validation_data_dir = "data/validation"
test_data_dir = "data/test"
MODEL_SUMMARY_FILE = "model_summary.txt"

# hyperparameters
IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256
RGB = 3
GRAY = 1
EPOCHS = 20
BATCH_SIZE = 4

# create CNN model
model = Sequential()

model.add(Conv2D(32,(3,3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, RGB), activation='relu'))
model.add(Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3), padding='same', activation='relu'))
model.add(Conv2D(128,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), padding='same', activation='relu'))
model.add(Conv2D(256,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
model.summary()

with open(MODEL_SUMMARY_FILE,"w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

# data augmentation
training_data_generator = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=5,
    horizontal_flip=False)

validation_data_generator = ImageDataGenerator(
    rescale=1/225
)

test_data_generator = ImageDataGenerator(
    rescale=1/255
)

# load data using flow (stream of data)
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1,
    class_mode="categorical",
    shuffle=False)

# training
if(TRAIN):
    model.fit_generator(
        training_generator,
        steps_per_epoch = len(training_generator.filenames) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
        verbose=1,
        callbacks=[CSVLogger('log.csv', append=False, separator=",")])

    model.save('model.h5')