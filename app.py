import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import struct
import pandas as pd
import numpy as np
import ffmpeg
import pydub
from sqlalchemy.dialects import postgresql
import urllib.parse
import config
from sqlalchemy import engine
import tensorflow as tf
from io import StringIO
import wave
from scipy.io import wavfile
import soundfile as sf
import io
import cv2
import sounddevice as sd
from pydub import AudioSegment
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from scipy import stats
import scipy
from scipy.io.wavfile import read
import crepe
from scipy.io.wavfile import write
import soundfile as sf
import boto3
import subprocess
from botocore.client import Config
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from sklearn.preprocessing import normalize
from natsort import natsorted


s3_client = boto3.client(
    's3',
    region_name='ru-msk',
    aws_access_key_id='opupsuKWKbmnc6cAeoQubP',
    aws_secret_access_key='guUYUDdCAgAJp757thbFLaqd2Y9H7XefW8P6FkbwLcFM',
)

session = boto3.session.Session()
s3_client = session.client(
    service_name='s3',
    endpoint_url='https://hb.bizmrg.com',
    aws_access_key_id='opupsuKWKbmnc6cAeoQubP',
    aws_secret_access_key='guUYUDdCAgAJp757thbFLaqd2Y9H7XefW8P6FkbwLcFM'
)


fs = 44100
seconds = 30
scale_file = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

sf.write('input_file.wav', scale_file, fs)


y, sr = librosa.load('input_file.wav')
S = librosa.stft(y, center=False)
secs = librosa.get_duration(S=S, sr=sr)



secs = round(secs * 4)


count = 0
k = 0
df_file_paths = pd.DataFrame(columns=['Paths'])
names = []
for i in range(secs):
    t1 = count * 250
    t2 = (count + 1) * 250
    newAudio = AudioSegment.from_wav('input_file.wav')
    newAudio = newAudio[t1:t2]

    name = 'output.wav'
    newAudio.export(name, format="wav")
    char = str(k)
    test_bucket_name = 'splitedinputfiles'
    file_bucket_name = str('split_file' + char + '.wav')
    s3_client.upload_file('output.wav', test_bucket_name, file_bucket_name)


    names.append(names)

    k += 1
    count += 1


paginator = s3_client.get_paginator('list_objects')
result = paginator.paginate(Bucket=test_bucket_name)


file_paths_list = []

for page in result:
    if "Contents" in page:
        for key in page["Contents"]:
            keyString = key["Key"]
            # print(keyString)
            file_paths_list.append(keyString)


natsorted(file_paths_list)


url_object = sqlalchemy.engine.url.URL.create(
    "postgresql+psycopg2",
    username="Squirrel_Emma",
    password="U90y83tw523YEx5q",  # plain (unescaped) text
    host="85.192.32.89",
    database="PostgreSQL-7599",
)


engine = create_engine(url_object)
df_file_paths.to_sql(name='File_paths_table', con=engine, if_exists='replace')



def createSpectrogram_pitch(file_name):
    try:

        fig = plt.figure(figsize=[1.5, 10])
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

        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')

    except:

        print("Create spectrogram failure:", file_name)
        return None

    return mfccs_norm


spectrograms_pitch = []

for i in range(len(file_paths_list)):

    s3_client.download_file('splitedinputfiles', file_paths_list[i], 'newinput.wav')

    file_name = 'newinput.wav'

    data1 = createSpectrogram_pitch(file_name)
    spectrograms_pitch.append([data1])
    plt.close('all')
    if i % 10 == 0:
        print('current processing iteration', i)


global mfccs_norm
notes = []


def predict_pitch(input_audioFile):

    y, sr = librosa.load(input_audioFile)

    fig = plt.figure(figsize=[1.5, 10])

    conQfit = np.abs(librosa.cqt(y, sr=sr))

    librosa.display.specshow(conQfit, y_axis='cqt_note')


    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=(56 / 5))
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global mfccs_norm

    mfccs_norm = normalize(img, axis=0, norm='max')


    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')

    channels = 1
    row = 1
    spectrogram_shape1 = (row,) + mfccs_norm.shape + (channels,)
    x_reshape = np.array(i.reshape((spectrogram_shape1)) for i in mfccs_norm)

    global pitch_ETL_4d_output
    pitch_ETL_4d_output = mfccs_norm.reshape((spectrogram_shape1))
    print(pitch_ETL_4d_output.shape)

    s3_client.download_file('symphonicmasks', 'pitch_model.h5', 'pitch_model.h5')
    pitch_model = tf.keras.models.load_model('pitch_model.h5')


    pitch_result = pitch_model.predict(pitch_ETL_4d_output)

    pitch_scalar = np.argmax(pitch_result, axis=None, out=None)

    bucket_name = 'symphonicmasks'
    object_key = 'pitchName.csv'
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    pitch_Name_df = pd.read_csv(StringIO(csv_string))

    pitch_name_list = pitch_Name_df['0'].tolist()

    pitch_pred = pitch_name_list[pitch_scalar]

    notes.append(pitch_pred)


for i in range(len(file_paths_list)):

    s3_client.download_file('splitedinputfiles', file_paths_list[i], 'newinput.wav')
    input_audioFile = 'newinput.wav'

    predict_pitch(input_audioFile)

k = 0
final_names = []
for i in range(len(file_paths_list)):

    s3_client.download_file('splitedinputfiles', file_paths_list[i], 'newinput1.wav')

    file_name = 'newinput1.wav'


    sr, y = wavfile.read(file_name)  # загрузили файл
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

    s3_client.download_file('symphonicmasks', 'flutes_notes_frequencies.csv', 'flutes_notes_frequencies.csv')
    pitch_model = tf.keras.models.load_model('pitch_model.h5')

    df_notes = pd.read_csv('flutes_notes_frequencies.csv', delimiter=';')

    flute_items = df_notes['Frequency'].tolist()
    right_note = min(flute_items, key=lambda x: abs(x - the_closet_note))
    new_fr = abs(right_note - the_closet_note)


    if the_closet_note > right_note:
        new1_fr = (right_note + new_fr) / right_note
        sr = round(44100 * new1_fr)
    elif the_closet_note < right_note:
        new1_fr = (right_note - new_fr) / right_note
        sr = round(44100 * new_fr)
    else:
        sr=sr

    duration = librosa.get_duration(filename=file_name)
    y, sr = librosa.load(file_name)
    if duration != 0.25:
        if duration < 0.25:
            x = duration
            x1 = 0.25 - duration
            x2 = 0.25 / x
            nnew = librosa.effects.time_stretch(y, rate=x * x2)

        elif duration > 0.25:
            x = duration - 0.25
            x1 = x / 0.682545453453
            nnew = librosa.effects.time_stretch(y, rate=x * x1)
        fs = sr
    else:
        nnew=y

    name = 'output_final.wav'
    newAudio = write(name, sr, nnew)

    char = str(k)
    test_bucket_name = 'newsplitedfiles'
    file_bucket_name = str('newsplitedfiles' + char + '.wav')
    s3_client.upload_file('output_final.wav', test_bucket_name, file_bucket_name)


    final_names.append(file_bucket_name)


    k += 1
    
combined_wav = AudioSegment.empty()                                                          
for i in range(len(final_names)):
    
    s3_client.download_file('newsplitedfiles',final_names[i], 'joinedoutput.wav')
    
    file_name = 'joinedoutput.wav'
    

    order = AudioSegment.from_wav(file_name) 
    combined_wav += order

combined_wav.export("newinput1.wav", format="wav")





