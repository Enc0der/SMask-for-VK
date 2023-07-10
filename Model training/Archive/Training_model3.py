import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
# Import dependencies
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
import json


# read data from Postgres Database
with open('settings.json') as settings_file:
    config = json.load(settings_file)
    print(config)

url_object = sqlalchemy.engine.url.URL.create(

    username=config['db_connection']['username'],
    password=config['db_connection']['password'],
    host='pgsql',
    database=config['db_connection']['database'],
    drivername=config['db_connection']['driver'],
)
# read data from Postgres Database
engine = create_engine(url_object)

notes_df = pd.read_sql_table('Notes_Spectrogram_Table', engine)
# notes_df

# Convert Spectrograms from list to ndarray
notes_df['Spectrogram'] = notes_df['Spectrogram'].apply(lambda x: np.array(x))

type(notes_df['Spectrogram'].iloc[0])


# Create train and test data sets
X_series = notes_df["Spectrogram"]
y = notes_df["Note"]

# Parameters
channels = 1  # number of audio channels
spectrogram_shape = X_series[1].shape + (channels,)
batch = spectrogram_shape[1]
# X_series.shape
# (2913,)

count = 0

for i in X_series[1]:
    count += 1


# Reshape X into size of spectrogram and convert to ndarray
X = np.array([i.reshape(spectrogram_shape) for i in X_series])

# Encode pitches

le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Reshape for model
# X_train = np.array([X.reshape(20, 20, 1) for x in X_train])
# X_test = np.array([X.reshape(20, 20, 1) for x in X_test])

# onehotencoder = OneHotEncoder()
y_test_hot = to_categorical(y_test)
y_train_hot = to_categorical(y_train)

# Troubleshooting queries
# type(X_train[1])
# X_train[1].shape
# X_train[1]
# (22, 128, 1)

# # Model
# model = Sequential()
# # input_shape=(128, 128, 1)

# model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=spectrogram_shape))
# model.add(MaxPooling2D((4, 2), strides=(4, 2)))
# model.add(Activation('relu'))

# model.add(Conv2D(48, (5, 5), padding="valid"))
# model.add(MaxPooling2D((4, 2), strides=(4, 2)))
# model.add(Activation('relu'))

# model.add(Conv2D(48, (5, 5), padding="valid"))
# model.add(Activation('relu'))

# model.add(Flatten())
# model.add(Dropout(rate=0.5))

# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.5))

# model.add(Dense(10))
# model.add(Activation('softmax'))

# model.compile(
#   optimizer="Adam",
#   loss="categorical_crossentropy",
#   metrics=['accuracy'])

# model.fit(
#   x=X_train,
#   y=y_train,
#     epochs=12,
#     batch_size=batch,
#     validation_data= (X_test, y_test))

# score = model.evaluate(
#   x=X_test,
#   y=y_test)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# Model
notes_model = Sequential()
notes_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=spectrogram_shape, padding='same'))
notes_model.add(LeakyReLU(alpha=0.1))
notes_model.add(MaxPooling2D((2, 2), padding='same'))


notes_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
notes_model.add(LeakyReLU(alpha=0.1))
notes_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


notes_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
notes_model.add(LeakyReLU(alpha=0.1))
notes_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


notes_model.add(Flatten())
notes_model.add(Dense(128, activation='linear'))
notes_model.add(LeakyReLU(alpha=0.1))
notes_model.add(Dense(12, activation='softmax'))

notes_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# notes_model.summary()

notes_train = notes_model.fit(X_train, y_train_hot, batch_size=batch, epochs=8, verbose=1, validation_data=(X_test, y_test_hot))
