import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import struct
import pandas as pd
import numpy as np
import ffmpeg


import wave
from scipy.io import wavfile
import soundfile as sf


import io
import cv2


from PIL import Image

from sklearn.preprocessing import normalize

# Извлекаем аудиофайлы для обучения и конвертируем их в спектрограммы для  обучения модели:


# готовим датасеты
df_local_file_paths = pd.read_csv('/Users/monglels/Desktop/Local_File_Path.csv', delimiter=';')
df_local_file_paths

df_local_file_paths['Local_File_Path'] = "/Users/monglels/Desktop/TinySOL" + '/' + df_local_file_paths[
    'Local_File_Path'].astype(str)
df_local_file_paths['Local_File_Path'] = + df_local_file_paths['Local_File_Path'].astype(str)

df_local_file_paths.head(10)

file_name = df_local_file_paths.loc[990, "Local_File_Path"]
file_name

# тестируем
import matplotlib

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=[1.5, 10])  # shorten the x-axis and focus on the y-axis

# Конвертируем массив в 'Constant-Q transform'
y, sr = librosa.load('/Users/monglels/Desktop/TinySOL/Keyboards/Accordion/ordinario/Acc-ord-C#3-mf-alt1-N.wav')

conQ_spec = np.abs(librosa.cqt(y, sr=sr))

librosa.display.specshow(conQ_spec, y_axis='cqt_note')


# Создаем функцию для конвертирования в спектрограмму:
def createSpectrogram_pitch(file_name):
    try:
        # Загружаем аудио из локаьной папки. С помощью librosa они автоматически конвертируются в sr = 22.05KHz
        # нормализуем bit-depth (-1 to 1) и сводим каналы в один канал.
        # Задаем размер рисунка
        fig = plt.figure(figsize=[1.5, 10])
        y, sr = sf.read(file_name, dtype='float32')

        # Конвертируем аудио массив в 'Constant-Q transform'.

        conQ_spec = np.abs(librosa.cqt(y, sr=sr))

        librosa.display.specshow(conQ_spec, y_axis='cqt_note')

        # конвертируем в 2D массив
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=(56 / 5))
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # нормализуем
        mfccs_norm = normalize(img, axis=0, norm='max')

        # закрываем изображение, чтобы оно не висело в течение цикла
        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')

    except:
    # Выводим это сообщение если загрузить файлы не удалось:

        print("Create spectrogram failure:", file_name)
        return None

    # Возвращаем спектрограмму


    return mfccs_norm

# прогоняем через цикл все файлы:
spectrograms_pitch = []

for i in range(len(df_local_file_paths)):

    file_name = df_local_file_paths.loc[i, "Local_File_Path"]
    data1 = createSpectrogram_pitch(file_name)
    spectrograms_pitch.append([data1])
    plt.close('all')
    if i % 100 == 0:
        print('current processing iteration', i)

spectrograms_pitch

# Готовим таблицы для обучения:


import soundfile as sf
import io
from six.moves.urllib.request import urlopen
from sklearn.preprocessing import normalize

# создаем датафрейм, в который сохраним значения спектрограмм в виде 2D массива
pitch_df = pd.DataFrame(spectrograms_pitch, columns=['Spectrogram'])
pitch_df

# открываем оригинальный датафрейм TINYSOL
tiny_soldf = pd.read_csv('/Users/monglels/Desktop/TinySOL/TinySOL_metadata.csv')

tiny_soldf

tiny_soldf_sample = pd.DataFrame(tiny_soldf)

# добавляем к нему столбец 'Note' и заполняем его названием соответствующей ноты(копируем из колонки 'Pitch')
import re

tiny_soldf_sample['Note'] = tiny_soldf_sample['Pitch'].str.rsplit('\d', expand=True)

tiny_soldf_sample

# убираем из значения колонки 'Note' цифру, отвечающую за октаву, и оставляем в ней только название ноты
tiny_soldf_sample['Note'] = tiny_soldf_sample['Note'].str[:-1]

tiny_soldf_sample

# Добавляем колонку 'Octave', куда переносим соответствующий номер октавы
tiny_soldf_sample['Octave'] = tiny_soldf_sample['Pitch'].str.extract('(\d)', expand=True)

tiny_soldf_sample

df_merge_col = pd.merge(df_local_file_paths, tiny_soldf_sample, right_index=True, left_index=True)
pitchDf_merged = pd.merge(pitch_df, df_merge_col, right_index=True, left_index=True)

# Удаляем ненужные столбцы
pitchDF_Final = pitchDf_merged.drop(['Path',
                                     # 'S3 file path', # Only use when pulling from S3 bucket
                                     'Fold',
                                     'Family',
                                     'Pitch ID',
                                     'Instrument (abbr.)',
                                     'Instrument (in full)',
                                     'Technique (abbr.)',
                                     'Technique (in full)',
                                     'Dynamics',
                                     'Dynamics ID',
                                     'Instance ID',
                                     'String ID (if applicable)',
                                     'Needed digital retuning'],
                                    axis=1)

# Меняем имена столбцов чтобы загрузить их в postgres


pitchDF_Final.rename(columns={'Local file path_y': 'File_Path'}, inplace=True)

# Сохраняем ноты и соответствующие им спектрограммы в датафрейме 'pitchDF_Final'
pitchDF_Final.to_csv('/Users/monglels/Desktop/TinySOL/pitchdf.csv')

pitchDF_Final

# Загружаем таблицы в CNN:


import psycopg2

# Функция помогает  postgres читать столбец спектрограммы в датафрейме
from psycopg2.extensions import register_adapter, AsIs


def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


psycopg2.extensions.register_adapter(np.ndarray, psycopg2._psycopg.AsIs)
psycopg2.extensions.register_adapter(np.float32, addapt_numpy_float32)

import sqlalchemy
from sqlalchemy import create_engine
import config
import psycopg2
# from config import db_password


# Импортируем эту колонку для упрощения чтения спектрограм
from sqlalchemy.dialects import postgresql

import urllib.parse

urllib.parse.quote_plus('U90y83tw523YEx5q')

from sqlalchemy import URL

url_object = URL.create(
    "postgresql+psycopg2",
    username="Squirrel_Emma",
    password="U90y83tw523YEx5q",
    host="85.192.32.89",
    database="PostgreSQL-7599",
)

from sqlalchemy import create_engine

engine = create_engine(url_object)

# In[55]:


# Upload the dataframes into the postgres as a table

# Connect to postgres please note you need to use your username and postgres pass you can save your pass in the confige file

# engine = create_engine('postgresql+psycopg2://postgres:' + db_password +'@localhost:5432/InstroPitch_DB')
# Connect to postgres please note you need to use your username and postgres pass you can save your pass in the confige file
engine = create_engine(url_object)

# Создаем Notes датафрейм как Table в Postgres
pitchDF_Final.to_sql('Pitch_Spectrogram_Table', engine, if_exists='replace',
                     ######use relace if tables are already in your DB
                     method=None, dtype={'Spectrogram': postgresql.ARRAY(sqlalchemy.types.REAL, dimensions=2)}
                     )

# Создаем tables в postgres для Original Tables

orig_note_df = pd.read_csv('/Users/monglels/Desktop/TinySOL/Orig_notes_table.csv')

pitch_df = pd.read_csv('/Users/monglels/Desktop/TinySOL/Pitch_table.csv')

notes_freq_df = pd.read_csv('/Users/monglels/Desktop/TinySOL/Notes_frequency.csv')

# загружаем основную информацию по нотам в ДатаБазу
orig_note_df.to_sql(name='Notes_table', con=engine, if_exists='replace')

# загружаем основную информацию по высоте в ДатаБазу
pitch_df.to_sql(name='Pitch_table', con=engine, if_exists='replace')

# загружаем частоту нот/октав в ДатаБазу
notes_freq_df.to_sql(name='Notes_frequency_table', con=engine, if_exists='replace')

# Загружаем Database Tables в Python и тренируем модель:


import sqlalchemy
from sqlalchemy import create_engine
# from config import db_password_susie
import pandas as pd

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.utils import to_categorical
# from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt
import numpy as np

from sqlalchemy import URL

url_object1 = URL.create(
    "postgresql",
    username="Squirrel_Emma",
    password="U90y83tw523YEx5q",
    host="85.192.32.89",
    database="PostgreSQL-7599",
)

# читаем информацию с Postgres Database
engine = create_engine(url_object1)

notes_df = pd.read_sql_table('Pitch_Spectrogram_Table', engine)

# Конвертируем спектрограму из списка в массив
notes_df['Spectrogram'] = notes_df['Spectrogram'].apply(lambda x: np.array(x))

type(notes_df['Spectrogram'].iloc[0])

notes_df.head(5)

# Создаем train and test data sets
X_series = notes_df["Spectrogram"]
y = notes_df["Pitch"]

X_series.head(5)

channels = 1  # количество аудиоканалов
spectrogram_shape = X_series[1].shape + (channels,)

# spectrogram_shape = X_series[1].shape
batch = spectrogram_shape[1]

batch

X = np.array([i.reshape((spectrogram_shape)) for i in X_series])

X

le = LabelEncoder()
y = le.fit_transform(y)

# Разделяем датасеты
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=83)

y_test_hot = to_categorical(y_test)
y_train_hot = to_categorical(y_train)

# сохраняем названия определенных высот
CV_pitch_name = le.classes_
CV_pitch_df = pd.DataFrame(CV_pitch_name)
CV_pitch_df
CV_pitch_df.to_csv('/Users/monglels/Desktop/TinySOL/pitchName.csv')

type(X_train[1])
X_train[1].shape

pitch_model = Sequential()

pitch_model.add(Conv2D(24, kernel_size=(3, 3), activation='linear', input_shape=(spectrogram_shape), padding='same'))
pitch_model.add(LeakyReLU(alpha=0.1))
pitch_model.add(MaxPooling2D((2, 2), padding='same'))

pitch_model.add(Conv2D(48, (3, 3), activation='linear', padding='same'))
pitch_model.add(LeakyReLU(alpha=0.1))
pitch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

pitch_model.add(Conv2D(96, (3, 3), activation='linear', padding='same'))
pitch_model.add(LeakyReLU(alpha=0.1))
pitch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

pitch_model.add(Flatten())
pitch_model.add(Dense(112, activation='linear'))

pitch_model.add(LeakyReLU(alpha=0.1))
pitch_model.add(Dense(82, activation='softmax'))  ########### make 14 variable for instrument num

pitch_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

pitch_model.summary()

pitch_train = pitch_model.fit(X_train, y_train_hot, batch_size=batch, epochs=10, verbose=1,
                              validation_data=(X_test, y_test_hot))

pitch_train

# рисуем метрику
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(pitch_train.history['accuracy'])
plt.plot(pitch_train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(pitch_train.history['loss'])
plt.plot(pitch_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
fig

# получаем предсказание
pred = pitch_model.predict(X_test)

y_pred = [np.argmax(y, axis=None, out=None) for y in pred]
y_actual = [np.argmax(y, axis=None, out=None) for y in y_test_hot]

Pitch_results_df = pd.DataFrame({'Pred': y_pred, 'Actual': y_actual})
Pitch_results_df

# Экспортируем модель в HDF5 файл

# сохраняем тренированную модель
pitch_model.save("/Users/monglels/Desktop/TinySOL/Result_models/pitch_model.h5")

import pickle

with open('/Users/monglels/Desktop/TinySOL/Result_models/PKL_trained_pitch_model.pkl', 'wb') as pitch_model_file:
    pickle.dump(pitch_model, pitch_model_file)

# в датафрейм с адресами входящих мини-файлов добавляем столбец с соответствующей им ноте


df_file_paths.insert(loc=len(df_file_paths.columns), column='trying_note')

df_file_paths

# Далее см. код для пользователя


