import soundfile as sf
import io
import cv2
from sklearn.preprocessing import normalize
from psycopg2.extensions import AsIs
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from sqlalchemy.dialects import postgresql
import urllib.parse
from keras.layers import LeakyReLU
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import pickle
import librosa
import matplotlib as plt

df_local_file_paths = pd.read_csv('/Users/monglels/Desktop/Local_File_Path.csv',
                                  delimiter=';')

df_local_file_paths['Local_File_Path'] = "/Users/monglels/Desktop/TinySOL" + '/' + df_local_file_paths[
    'Local_File_Path'].astype(str)
df_local_file_paths['Local_File_Path'] = + df_local_file_paths['Local_File_Path'].astype(str)

file_name = df_local_file_paths.loc[990, "Local_File_Path"]
file_name

fig = plt.figure(figsize=[1.5, 10])  # shorten the x-axis and focus on the y-axis

y, sr = librosa.load('/Users/monglels/Desktop/TinySOL/Keyboards/Accordion/ordinario/Acc-ord-C#3-mf-alt1-N.wav')

conQ_spec = np.abs(librosa.cqt(y, sr=sr))

librosa.display.specshow(conQ_spec, y_axis='cqt_note')


# Обучаем модель:

# Создаем функцию для конвертирования в спектрограмму:
def createSpectrogram_pitch(file_name):

    fig = plt.figure(figsize=[1.5, 10])
    y, sr = sf.read(file_name, dtype='float32')

    # Конвертируем аудио массив в 'Constant-Q transform'. 86 bins are created to take pitches form C1 to C#8
    conQ_spec = np.abs(librosa.cqt(y, sr=sr))
    librosa.display.specshow(conQ_spec, y_axis='cqt_note')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=(56 / 5))
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # нормализуем
    mfccs_norm = normalize(img, axis=0, norm='max')
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

# создаем датафрейм, в который сохраним значения спектрограмм в виде 2D массива
pitch_df = pd.DataFrame(spectrograms_pitch, columns=['Spectrogram'])
pitch_df

# открываем оригинальный датафрейм TINYSOL
tiny_soldf = pd.read_csv('/Users/monglels/Desktop/TinySOL/TinySOL_metadata.csv')

tiny_soldf_sample = pd.DataFrame(tiny_soldf)

# добавляем к нему столбец 'Note' и заполняем его названием соответствующей ноты(копируем из колонки 'Pitch')
tiny_soldf_sample['Note'] = tiny_soldf_sample['Pitch'].str.rsplit('\d', expand=True)

# убираем из значения колонки 'Note' цифру, отвечающую за октаву, и оставляем в ней только название ноты
tiny_soldf_sample['Note'] = tiny_soldf_sample['Note'].str[:-1]

# Добавляем колонку 'Octave', куда переносим соответствующий номер октавы
tiny_soldf_sample['Octave'] = tiny_soldf_sample['Pitch'].str.extract('(\d)', expand=True)

df_merge_col = pd.merge(df_local_file_paths, tiny_soldf_sample, right_index=True, left_index=True)
pitchDf_merged = pd.merge(pitch_df, df_merge_col, right_index=True, left_index=True)

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

pitchDF_Final.rename(columns={'Local file path_y': 'File_Path'}, inplace=True)
pitchDF_Final.to_csv('/Users/monglels/Desktop/TinySOL/pitchdf.csv')


def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


psycopg2.extensions.register_adapter(np.ndarray, psycopg2._psycopg.AsIs)
psycopg2.extensions.register_adapter(np.float32, addapt_numpy_float32)

# Импортируем эту колонку для упрощения чтения спектрограм

urllib.parse.quote_plus('U90y83tw523YEx5q')

url_object = sqlalchemy.engine.url.URL.create(
    "postgresql+psycopg2",
    username="Squirrel_Emma",
    password="U90y83tw523YEx5q",  # plain (unescaped) text
    host="85.192.32.89",
    database="PostgreSQL-7599",
)

engine = create_engine(url_object)

pitchDF_Final.to_sql('Pitch_Spectrogram_Table', engine, if_exists='replace',

                     method=None, dtype={'Spectrogram': postgresql.ARRAY(sqlalchemy.types.REAL, dimensions=2)}
                     )

orig_note_df = pd.read_csv('/Users/monglels/Desktop/TinySOL/Orig_notes_table.csv')
pitch_df = pd.read_csv('/Users/monglels/Desktop/TinySOL/Pitch_table.csv')
notes_freq_df = pd.read_csv('/Users/monglels/Desktop/TinySOL/Notes_frequency.csv')

orig_note_df.to_sql(name='Notes_table', con=engine, if_exists='replace')

pitch_df.to_sql(name='Pitch_table', con=engine, if_exists='replace')

notes_freq_df.to_sql(name='Notes_frequency_table', con=engine, if_exists='replace')

#  Загружаем Database Tables в Python и тренируем модель:

# читаем информацию с Postgres Database
engine = create_engine(url_object)

notes_df = pd.read_sql_table('Pitch_Spectrogram_Table', engine)

# Конвертируем спектрограму из списка в ndarray
notes_df['Spectrogram'] = notes_df['Spectrogram'].apply(lambda x: np.array(x))

type(notes_df['Spectrogram'].iloc[0])

# Создаем train and test data sets
X_series = notes_df["Spectrogram"]
y = notes_df["Pitch"]

channels = 1
spectrogram_shape = X_series[1].shape + (channels,)

batch = spectrogram_shape[1]

X = np.array([i.reshape((spectrogram_shape)) for i in X_series])

le = LabelEncoder()
y = le.fit_transform(y)

# Разделяем датасеты
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=83)

y_test_hot = to_categorical(y_test)
y_train_hot = to_categorical(y_train)

CV_pitch_name = le.classes_
CV_pitch_df = pd.DataFrame(CV_pitch_name)

CV_pitch_df.to_csv('/Users/monglels/Desktop/TinySOL/pitchName.csv')

pitch_model = Sequential()

pitch_model.add(Conv2D(24, kernel_size=(3, 3), activation='linear', input_shape=(spectrogram_shape), padding='same'))
pitch_model.add(LeakyReLU(alpha=0.1))
pitch_model.add(MaxPooling2D((2, 2), padding='same'))

pitch_model.add(Conv2D(48, (3, 3), activation='linear', padding='same'))
pitch_model.add(LeakyReLU(alpha=0.01))
pitch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

pitch_model.add(Conv2D(96, (3, 3), activation='linear', padding='same'))
pitch_model.add(LeakyReLU(alpha=0.01))
pitch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

pitch_model.add(Flatten())
pitch_model.add(Dense(112, activation='linear'))
pitch_model.add(LeakyReLU(alpha=0.1))
pitch_model.add(Dense(82, activation='softmax'))

pitch_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

pitch_model.summary()

pitch_train = pitch_model.fit(X_train, y_train_hot, batch_size=batch, epochs=10, verbose=1,
                              validation_data=(X_test, y_test_hot))

# получаем предсказание
pred = pitch_model.predict(X_test)
# reverse to_categorical function
y_pred = [np.argmax(y, axis=None, out=None) for y in pred]
y_actual = [np.argmax(y, axis=None, out=None) for y in y_test_hot]

Pitch_results_df = pd.DataFrame({'Pred': y_pred, 'Actual': y_actual})
Pitch_results_df

# сохраняем тренированную модель
pitch_model.save("/Users/monglels/Desktop/TinySOL/Result_models/pitch_model.h5")

# save in binary format,wb replace old file with new one
with open('/Users/monglels/Desktop/TinySOL/Result_models/PKL_trained_pitch_model.pkl', 'wb') as pitch_model_file:
    pickle.dump(pitch_model, pitch_model_file)
