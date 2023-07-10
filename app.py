"""
This script processes videos sent via Telegram, extracts the audio, splits it into smaller segments,
performs pitch analysis on each segment, generates a modified audio file with adjusted pitch,
and combines the modified audio with the original video to create a final video with modified audio.
"""

import json
import moviepy.editor as mp
from scipy.io.wavfile import write
import boto3
import statistics
import librosa
import pandas as pd
from io import StringIO
import telebot
# import config
from pydub import AudioSegment
import cv2
from sklearn.preprocessing import normalize
from natsort import natsorted
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tensorflow as tf
import io
import matplotlib
from moviepy.editor import VideoFileClip, AudioFileClip
from config import TOKEN

# Load configuration settings from 'settings.json'
with open('settings.json') as settings_file:
    config = json.load(settings_file)

# Create an S3 client with provided credentials
s3_client = boto3.client(
    's3',
    region_name=config['s3_client']['region_name'],
    aws_access_key_id=config['s3_client']['aws_access_key_id'],
    aws_secret_access_key=config['s3_client']['aws_secret_access_key'],
    endpoint_url=config['s3_client']['endpoint_url']
)

# Set the backend for matplotlib
matplotlib.pyplot.switch_backend('Agg')

# Load Telegram bot token from config
bot = telebot.TeleBot(TOKEN)

# Create a boto3 session
session = boto3.session.Session()

final_names = []
notes = []
notes_list = []
file_paths_list = []
count = 1

# Download pitch model from S3
s3_client.download_file('symphonicmasks', 'pitch_model.h5', 'pitch_model.h5')
pitch_model = tf.keras.models.load_model('pitch_model.h5')


@bot.message_handler(content_types=['video'])
def get_file(message):
    """
    Telegram bot handler for processing video files.

    Args:
        message: Telegram message object containing the video file.
    """
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
    my_clip.audio.write_audiofile(r'input_audio_file.wav')
    my_clip.write_videofile(r'input_video_file.mp4')
    prepare_file()


def prepare_file():
    """
    Prepare the input audio file by splitting it into smaller segments.

    Uses librosa to load the audio, split it into segments of 250ms each,
    and upload the segments to an S3 bucket for further processing.
    """
    y, sr = librosa.load(r'input_audio_file.wav')
    S = librosa.stft(y, center=False)
    secs = librosa.get_duration(S=S, sr=sr)
    secs = round(secs * 4)
    count_prepare_file = 0
    k = 0
    names = []
    for i in range(secs):
        t1 = count_prepare_file * 250   # Works in milliseconds
        t2 = (count_prepare_file + 1) * 250  # мин 16е в темпе 60
        newAudio = AudioSegment.from_wav(r'input_audio_file.wav')
        newAudio = newAudio[t1:t2]
        name = 'output.wav'
        newAudio.export(name, format="wav")
        char = str(k)
        test_bucket_name = 'splitedinputfiles'
        file_bucket_name = str('split_file' + char + '.wav')
        s3_client.upload_file('output.wav', test_bucket_name, file_bucket_name)

        names.append(names)

        k += 1
        count_prepare_file += 1

    paginator = s3_client.get_paginator('list_objects')
    global result
    result = paginator.paginate(Bucket=test_bucket_name)
    create_file_paths_list()


def create_file_paths_list():
    """
    Create a list of file paths in the S3 bucket containing the split audio segments.
    """
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

        s3_client.download_file('splitedinputfiles',
                                file_paths_list[i], 'origin_input.wav')
        input_audiofile = 'origin_input.wav'

        if i < len(file_paths_list):
            predict_pitch(input_audiofile)


def predict_pitch(input_audiofile):
    """
    Perform pitch analysis on the given audio file segment.

    Uses librosa and a pre-trained pitch prediction model to analyze the pitch of the audio.
    Saves the analysis results and generates a spectrogram image for visualization.

    Args:
        input_audiofile: Path to the audio file segment to analyze.
    """
    global notes
    y, sr = librosa.load(input_audiofile)
    fig = plt.figure(figsize=[1.5, 10])

    conqfit = np.abs(librosa.cqt(y, sr=sr))

    librosa.display.specshow(conqfit, y_axis='cqt_note')

    pitches, magnitudes = librosa.core.piptrack(y=y,
                                                sr=sr, fmin=250, fmax=2000)
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

    channels = 1
    row = 1
    spectrogram_shape1 = (row,) + mfccs_norm.shape + (channels,)
    np.array(i.reshape((spectrogram_shape1)) for i in mfccs_norm)
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

    """
    Downloads audio files, performs processing, and combines them with a video file.
    Finally, sends the finished video to a chatbot.
    """
    k = 0
    final_names = []
    file_paths_list = []  # Assuming this list is defined somewhere in the code

    for i in range(len(file_paths_list)):
        # Download the audio file
        s3_client.download_file('splitedinputfiles', file_paths_list[i], 'origin_input.wav')

        # Load the audio file
        y, sr = librosa.load('origin_input.wav')

        # Download the flute notes frequencies
        s3_client.download_file('symphonicmasks', 'flutes_notes_frequencies.csv', 'flutes_notes_frequencies.csv')

        # Read the flute notes frequencies file
        df_notes = pd.read_csv('flutes_notes_frequencies.csv', delimiter=';')

        # Find the closest note frequency
        the_closet_note = notes_list[i]  # Assuming notes_list is defined somewhere in the code
        flute_items = df_notes['Frequency'].tolist()
        right_note = min(flute_items, key=lambda x: abs(x - the_closet_note))
        new_fr = abs(right_note - the_closet_note)

        # Adjust the sample rate based on the closest note
        if the_closet_note > right_note:
            new1_fr = (right_note - new_fr) / right_note
            sr = round(22050 * new1_fr)
        elif the_closet_note < right_note:
            new1_fr = (right_note + new_fr) / right_note
            sr = round(22050 * new1_fr)
        else:
            sr = sr

        # Write the filtered audio file
        char = str(k)
        name = str('filtred_splited_files' + char + '.wav')
        write(name, sr, y)
        k += 1
        final_names.append(name)

    # Combine the filtered audio files
    combined_wav = AudioSegment.empty()
    for i in range(len(final_names)):
        order = AudioSegment.from_wav(final_names[i])
        combined_wav += order

    # Export the combined audio as WAV
    combined_wav.export("final_audio.wav", format="wav")

    # Load the input video file
    video_clip = VideoFileClip('input_video_file.mp4')

    # Load the final audio file
    audio_clip = AudioFileClip('final_audio.wav')

    # Set the audio of the video clip to the final audio
    final_clip = video_clip.set_audio(audio_clip)

    # Write the final video file
    final_clip.write_videofile('finished_video.mp4')

    # Open the finished video file
    video = open(r'finished_video.mp4', 'rb')

    # Send the video to the chatbot
    bot.send_video(mesglob.from_user.id, video)

    # Close the video file
    video.close()


# Start the bot's polling
bot.infinity_polling()

