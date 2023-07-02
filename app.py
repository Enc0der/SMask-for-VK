import moviepy.editor as mp
from scipy import stats
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import boto3
import librosa
import struct
import moviepy.editor as mpe
import pandas as pd
import ffmpeg
from io import StringIO
import telebot
import datetime
from sqlalchemy import create_engine
import config
from scipy import stats
import scipy
import urllib.parse
urllib.parse.quote_plus('U90y83tw523YEx5q')
import psycopg2
from sqlalchemy import engine
import statistics
import sounddevice as sd
from pydub import AudioSegment
import cv2
from PIL import Image
import subprocess
from botocore.client import Config
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from sklearn.preprocessing import normalize
from natsort import natsorted
import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pydub
from sqlalchemy.dialects import postgresql
import tensorflow as tf
import wave
import soundfile as sf
import io
import sqlalchemy
import matplotlib


matplotlib.pyplot.switch_backend('Agg')

bot = telebot.TeleBot(config.token)

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


final_names = []
notes = []
spectrograms_pitch = []
notes_list = []
video_path = 'char'
audio_path = 'char'
file_paths_list = []
count = 1


s3_client.download_file('symphonicmasks', 'pitch_model.h5', 'pitch_model.h5')
pitch_model = tf.keras.models.load_model('pitch_model.h5')


@bot.message_handler(content_types=['video'])
def get_file(message):
    global file_name
    file_name = message.json['video']['file_name']
    file_info = bot.get_file(message.video.file_id)
    with open(file_name, "wb") as f:
        file_content = bot.download_file(file_info.file_path)
        f.write(file_content)
    global mesglob
    mesglob = message
    bot.reply_to(message, f"Привет! Ого, какое шикарное исполнение! "
                          f" Уже готовлю видео, это займет меньше минуты...")
    my_clip = mp.VideoFileClip(file_name)
    audio_path = my_clip.audio.write_audiofile(r'input_audio_file.wav')
    video_path = my_clip.write_videofile(r'input_video_file.mp4')
    prepare_file()


def prepare_file():
    y, sr = librosa.load(r'input_audio_file.wav')
    S = librosa.stft(y, center=False)
    secs = librosa.get_duration(S=S, sr=sr)
    secs = round(secs * 4)
    count_prepare_file = 0
    k = 0
    names = []
    for i in range(secs):
        t1 = count_prepare_file * 250 # Works in milliseconds
        t2 = (count_prepare_file + 1) * 250  # минимальное деление- 16е в темпе 60
        newAudio = AudioSegment.from_wav(r'input_audio_file.wav')
        newAudio = newAudio[t1:t2]
        # name= '/Users/monglels/Desktop/SymphonicMasks/Splited_input_files/newSong'
        # for k in range(secs):
        name = 'output.wav'
        newAudio.export(name, format="wav")
        char = str(k)
        test_bucket_name = 'splitedinputfiles'
        file_bucket_name = str('split_file' + char + '.wav')
        s3_client.upload_file('output.wav', test_bucket_name, file_bucket_name)

        names.append(names)

        k += 1
        count_prepare_file += 1

    # Use the paginator funtion to go access through the folder and files in the S3 bucket
    paginator = s3_client.get_paginator('list_objects')
    global result
    result = paginator.paginate(Bucket=test_bucket_name)
    create_file_paths_list()


def create_file_paths_list():
    # # Create for loop to go through all the .wav files path in the folders and add them to keylist list Icreated.
    global result
    global file_paths_list
    for page in result:
        if "Contents" in page:
            for key in page["Contents"]:
                keyString = key["Key"]
                # print(keyString)
                file_paths_list.append(keyString)


    file_paths_list = natsorted(file_paths_list)


    for i in range(len(file_paths_list)):

        s3_client.download_file('splitedinputfiles', file_paths_list[i], 'origin_input.wav')
        input_audiofile = 'origin_input.wav'

        if i < len(file_paths_list):
            predict_pitch(input_audiofile)


def predict_pitch(input_audiofile):

    global notes
    y, sr = librosa.load(input_audiofile)  # remove offset=length/6, duration=1, res_type='kaiser_fast'
    fig = plt.figure(figsize=[1.5, 10])
    # Convert audio array to 'Constant-Q transform'. 86 bins are created to take pitches form C1 to C#8
    conqfit = np.abs(librosa.cqt(y, sr=sr))

    librosa.display.specshow(conqfit, y_axis='cqt_note')

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=250, fmax=2000)
    # get indexes of the maximum value in each time slice
    max_indexes = np.argmax(magnitudes, axis=0)
    # get the pitches of the max indexes per time slice
    pitches = pitches[max_indexes, range(magnitudes.shape[1])]
    global frequency_note
    frequency_note = pitches
    frequency_note_list = frequency_note.tolist()

    global notes_list
    last_pitches = statistics.mean(frequency_note_list)
    notes_list.append(last_pitches)


    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=(56 / 5))
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global mfccs_norm
    # normalize
    mfccs_norm = normalize(img, axis=0, norm='max')
    #
    # plt.close()
    # fig.clf()
    # plt.close(fig)
    # plt.close('all')
    # convert mfccs_norm into 4d array
    channels = 1  # number of audio channels
    row = 1
    spectrogram_shape1 = (row,) + mfccs_norm.shape + (channels,)
    x_reshape = np.array(i.reshape((spectrogram_shape1)) for i in mfccs_norm)
    pitch_ETL_4d_output = mfccs_norm.reshape((spectrogram_shape1))

    pitch_result = pitch_model.predict(pitch_ETL_4d_output)

    # reverse to_categorical() function, get correlated pitch_name
    pitch_scalar = np.argmax(pitch_result, axis=None, out=None)

    bucket_name = 'symphonicmasks'
    object_key = 'pitchName.csv'
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    pitch_name_df = pd.read_csv(StringIO(csv_string))

    pitch_name_list = pitch_name_df['0'].tolist()

    pitch_pred = pitch_name_list[pitch_scalar]

    notes.append(pitch_pred)
    global count
    if count == len(file_paths_list):
        making_return_video()

    count += 1

def making_return_video():


    k = 0
    for i in range(len(file_paths_list)):

        s3_client.download_file('splitedinputfiles', file_paths_list[i], 'origin_input.wav')

        # file_name = 'newinput1.wav'
        y, sr = librosa.load('origin_input.wav')

        s3_client.download_file('symphonicmasks', 'flutes_notes_frequencies.csv', 'flutes_notes_frequencies.csv')
        # pitch_model = tf.keras.models.load_model('pitch_model.h5')

        df_notes = pd.read_csv('flutes_notes_frequencies.csv', delimiter=';')

        the_closet_note = notes_list[i]
        flute_items = df_notes['Frequency'].tolist()
        right_note = min(flute_items, key=lambda x: abs(x - the_closet_note))
        new_fr = abs(right_note - the_closet_note)

        if the_closet_note > right_note:
            new1_fr = (right_note - new_fr) / right_note
            sr = round(22050 * new1_fr)
        elif the_closet_note < right_note:
            new1_fr = (right_note + new_fr) / right_note
            sr = round(22050 * new1_fr)
        else:
            sr = sr

        char = str(k)
        name = str('filtred_ splited_files' + char + '.wav')
        newAudio = write(name, sr, y)
        k += 1
        final_names.append(name)
       

    combined_wav = AudioSegment.empty()
    for i in range(len(final_names)):
        order = AudioSegment.from_wav(final_names[i])
        combined_wav += order
    combined_wav.export("final_audio.wav", format="wav")
    audio = open(r'final_audio.wav', 'rb')
    bot.send_audio(mesglob.from_user.id, audio)
    audio.close()
# send_songs(message)
# ffmpeg.concat(video_path, combined_wav, v=1, a=1).output('./processed_folder/finished_video.mp4').run()
#input_video = ffmpeg.input("../resources/video_with_audio.mp4")
#added_audio = ffmpeg.input("../resources/dance_beat.ogg").audio.filter('adelay', "1500|1500")

# merged_audio = ffmpeg.filter([video_path, combined_wav], 'amix')
#
# (
#     ffmpeg
#     .concat(video_path, merged_audio, v=1, a=1)
#     .output("mix_delayed_audio.mp4")
#     .run(overwrite_output=True)
# )

# # video_path = input_video_file.mp4
# # audio_path = 'newinput1.wav'
# cmd = "ffmpeg -i {video_path} -i {audio_path} -c:v copy output.mp4"
# os.system(cmd)
# def send_songs(message):
#
# audio = open( 'newinput1.wav', 'rb')
# bot.send_chat_action(message.from_user.id, 'upload_audio')
# bot.send_audio(message.from_user.id, audio)
# audio.close()
# # video = open(cmd, 'rb')


# audio.close()
# bot.polling(non_stop=True)
bot.infinity_polling()
