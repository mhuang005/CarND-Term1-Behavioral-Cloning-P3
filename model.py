
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Lambda, Conv2D
from keras.layers import Dropout, Cropping2D
from keras.models import Sequential
from keras import backend as K

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import cv2


flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_integer('epochs', 20, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_float('correction', 0.15, "The correction on the left/right angles.")

# Read the csv data (only first 4 columns)
df = pd.read_csv('./data/driving_log.csv', usecols=[0,1,2,3])
cols = list(df) # i.e., ['center', 'left', 'right', 'steering']

# Combine the image names and steering angles to X_data, y_data
X_data = list(df[cols[0]])
y_data = list(df[cols[-1]])

X_data.extend(list(df[cols[1]]))
y_data.extend(list(df[cols[-1]] + FLAGS.correction))

X_data.extend(list(df[cols[2]]))
y_data.extend(list(df[cols[-1]] - FLAGS.correction))

# Split the data into the training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, train_size=0.8,
                                                      random_state=2018)


def batch_generator(data, angles, image_path, input_shape, batch_size):
    while True:
        # Shuffle the training data before each new epoch
        X, y = shuffle(data, angles)
        for offset in range(0, len(data), batch_size):
            y_batch = np.array(y[offset:offset+batch_size])
            X_batch = []
            for e in enumerate(X[offset:offset+batch_size]):
                image_name = image_path + e[1].split('/')[1]
                image = cv2.imread(image_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if (np.random.randint(2) == 0):
                    X_batch.append(image)
                else:  # flip the image
                    X_batch.append(cv2.flip(image, 1))
                    y_batch[e[0]] *= -1.0
            X_batch = np.array(X_batch)
            yield X_batch, y_batch
                    
                
input_shape = (160, 320, 3)
image_path = './data/IMG/'

# Define generators
train_generator = batch_generator(X_train, y_train, image_path,
                                  image_path, FLAGS.batch_size)
valid_generator = batch_generator(X_valid, y_valid, image_path,
                            image_path, FLAGS.batch_size)

# Model architecture -- A modified  NVIDIA architecture
# Reference: End to End Learning for Self-Driving Cars (NVIDIA's paper)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator= train_generator,
                    steps_per_epoch = len(X_train)//FLAGS.batch_size,
                    epochs = FLAGS.epochs,
                    verbose = 2,
                    validation_data = valid_generator,
                    validation_steps = len(X_valid)//FLAGS.batch_size)

# Save the model
model.save('model.h5')
print('Model saved.')

K.clear_session()




