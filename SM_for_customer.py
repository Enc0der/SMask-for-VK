fs = 44100  # Частота дискретизации
seconds = 30  # Продолжительность записи
scale_file = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Дождитесь окончания записи

sf.write('input_file.wav', scale_file, fs)

#  Делим его на мелкие файлы по милисекундно:


y, sr = librosa.load('/Users/monglels/Desktop/SymphonicMasks/test_file.wav')
S = librosa.stft(y, center=False)
secs = librosa.get_duration(S=S, sr=sr)

secs = round(secs * 4)
secs

from pydub import AudioSegment

count = 0
k = 0
df_file_paths = pd.DataFrame(columns=['Paths'])
names = []
for i in range(secs):
    t1 = count * 250  # Works in milliseconds
    t2 = (count + 1) * 250  # минимальное деление- 16е в темпе 60
    newAudio = AudioSegment.from_wav('/Users/monglels/Desktop/SymphonicMasks/test_file.wav')
    newAudio = newAudio[t1:t2]
    # name= '/Users/monglels/Desktop/SymphonicMasks/Splited_input_files/newSong'
    # for k in range(secs):

    char = str(k)
    name = str('/Users/monglels/Desktop/SymphonicMasks/Splited_input_files/newSong' + char + '.wav')
    newAudio.export(name, format="wav")
    names.append(name)

    k += 1
    count += 1

df_file_paths = pd.DataFrame(names, columns=['Paths'])

import sqlalchemy
from sqlalchemy import create_engine
import config
import psycopg2

from sqlalchemy.dialects import postgresql
import urllib.parse

urllib.parse.quote_plus('U90y83tw523YEx5q')
import sqlalchemy
from sqlalchemy import create_engine
import config
import psycopg2

from sqlalchemy import engine
# Import this library to help in reading the spectrogram column.
from sqlalchemy.dialects import postgresql
from sqlalchemy import URL

url_object = URL.create(
    "postgresql+psycopg2",
    username="Squirrel_Emma",
    password="U90y83tw523YEx5q",
    host="85.192.32.89",
    database="PostgreSQL-7599",
)
engine = create_engine(url_object)

# добавляем таблицу в Бакет

engine = create_engine(url_object)
df_file_paths.to_sql(name='File_paths_table', con=engine, if_exists='replace')


# Преобразовываем файлы в спектрограммы:


def createSpectrogram_pitch(file_name):
    try:

        fig = plt.figure(figsize=[1.5, 10])  # shorten the x-axis and focus on the y-axis
        y, sr = sf.read(file_name, dtype='float32')

        conQ_spec = np.abs(librosa.cqt(y, sr=sr))

        librosa.display.specshow(conQ_spec, y_axis='cqt_note')

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=(56 / 5))
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mfccs_norm = normalize(img, axis=0, norm='max')

        # закрываем изображения чтобы не показывались в течение цикла
        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')

    except:

        print("Create spectrogram failure:", file_name)
        return None

        # Возвращает спектрограмму
    return mfccs_norm


spectrograms_pitch = []

for i in range(len(df_file_paths)):

    # Указываем файл с адресами для функции спектрограммы
    file_name = df_file_paths.loc[i, "Paths"]

    data1 = createSpectrogram_pitch(file_name)
    spectrograms_pitch.append([data1])
    plt.close('all')
    if i % 10 == 0:
        print('current processing iteration', i)

spectrograms_pitch

#  Проводим через функцию все файлы:


'''This python script includes function, predict_pitch'''

# создаем функцию для предсказания высоты
global mfccs_norm
notes = []


def predict_pitch(input_audioFile):
    # используем  librosa чтобы конвертировать аудио файл в спектрограмму
    y, sr = librosa.load(input_audioFile)

    fig = plt.figure(figsize=[1.5, 10])
    # Конвертируем аудио массив в 'Constant-Q transform'.
    conQfit = np.abs(librosa.cqt(y, sr=sr))

    librosa.display.specshow(conQfit, y_axis='cqt_note')

    # Конвертируем изображение в 2D массив
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=(56 / 5))
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global mfccs_norm
    # нормализуем
    mfccs_norm = normalize(img, axis=0, norm='max')

    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    # конвертируем mfccs_norm в 4d массив
    channels = 1
    row = 1
    spectrogram_shape1 = (row,) + mfccs_norm.shape + (channels,)
    x_reshape = np.array(i.reshape((spectrogram_shape1)) for i in mfccs_norm)

    global pitch_ETL_4d_output
    pitch_ETL_4d_output = mfccs_norm.reshape((spectrogram_shape1))
    print(pitch_ETL_4d_output.shape)
    pitch_model = tf.keras.models.load_model('/Users/monglels/Desktop/TinySOL/Result_models/pitch_model.h5')

    # предсказываем

    pitch_result = pitch_model.predict(pitch_ETL_4d_output)

    pitch_scalar = np.argmax(pitch_result, axis=None, out=None)

    # вынимаем значени высоты из csv и трансформируем в список
    pitch_Name_df = pd.read_csv('/Users/monglels/Desktop/TinySOL/pitchName.csv')
    pitch_name_list = pitch_Name_df['0'].tolist()

    pitch_pred = pitch_name_list[pitch_scalar]

    pitch_model = tf.keras.models.load_model('/Users/monglels/Desktop/TinySOL/Result_models/pitch_model.h5')
    notes.append(pitch_pred)

    # загружаем тренированную модель


for i in range(secs):
    input_audioFile = df_file_paths.iloc[i]['Paths']
    predict_pitch(input_audioFile)

# Проводим через функцию все файлы:


# массив всейго файла:
notes

# Фильтруем каждый аудиофайл согласно требуемой ноте:


from scipy import stats
import scipy
from scipy.io.wavfile import read
import crepe
from scipy.io.wavfile import write

# перезаписываем оттюненые файлы
import soundfile as sf

k = 0
final_names = []
for i in range(len(df_file_paths)):
    file_path = df_file_paths.loc[i, 'Paths']

    sr, y = wavfile.read(file_path)  # загрузили файл
    time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)  # проходим его с crepe
    ee = time, frequency, confidence, activation
    df = pd.DataFrame(ee)
    df.to_csv('file2.csv', index=True
              , header=True)
    new = df.T
    new.columns = ['time', 'frequency', 'confidence', 'new_col4']
    new = new.drop(columns={'new_col4', 'confidence', 'time'})
    notes_new = new.drop(index=[0, 1, 2, 3, 4])
    the_closet_note = notes_new.mean()
    the_closet_note = int(the_closet_note)
    df_notes = pd.read_csv('/Users/monglels/Desktop/flutes_notes_frequencies.csv', delimiter=';')
    flute_items = df_notes['Frequency'].tolist()
    right_note = min(flute_items, key=lambda x: abs(x - the_closet_note))
    new_fr = abs(right_note - the_closet_note)

    if the_closet_note > right_note:
        new1_fr = (right_note + new_fr) / right_note
        sr = round(44100 * new1_fr)
    if the_closet_note < right_note:
        new1_fr = (right_note - new_fr) / right_note
        sr = round(44100 * new_fr)
    else:
        continue

    duration = librosa.get_duration(filename=file_path)
    if duration != 0.25:
        y, sr = librosa.load(file_path)

        if duration < 0.25:  # ускоряем
            x = duration
            x1 = 0.25 - duration
            x2 = 0.25 / x
            nnew = librosa.effects.time_stretch(y, rate=x * x2)

        if duration > 0.25:  # замедляем
            x = duration - 0.25
            x1 = x / 0.682545453453
            nnew = librosa.effects.time_stretch(y, rate=x * x1)
        fs = sr
    char = str(k)
    name = str('/Users/monglels/Desktop/SymphonicMasks/New_splitted_input_files/newSong' + char + '.wav')
    out_f = '/Users/monglels/Desktop/SymphonicMasks/newSong5656.wav'
    wavf.write(out_f, fs, nnew)

    sf.write(name, y, sr)
    final_names.append(name)

    df_new_file_paths = pd.DataFrame(final_names, columns=['Final_Paths'])

    k += 1

df_new_file_paths

# соединяем файлы
from pydub import AudioSegment

combined_wav = AudioSegment.empty()
for i in range(len(df_new_file_paths)):
    file_name = df_file_paths.loc[i, "Paths"]
    order = AudioSegment.from_wav(file_name)
    combined_wav += order

combined_wav.export("/Users/monglels/Desktop/SymphonicMasks/out.wav", format="wav")

