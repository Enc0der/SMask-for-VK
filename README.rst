======================
🎼🎻🎺 Symphonic Masks
======================

SymphonicMasks is a monophonic pitch redactor based on convolutional neural network operating on Q-transformed spectrograms form inputs audio-files.

Features:

* For training model I've used dataset TinySOL (these sounds were originally recorded at Ircam in Paris (France) between 1996 and 1999, as part of a larger project named Studio On Line (SOL))


  * ... and here is a dataset: `dataset
    <https://zenodo.org/record/3685367#.Xo1NVi2ZOuU>`_.


.. contents:: Main information


Running Symphonic Masks
--------

Import necessary libraries using pip install -r '.\requirements.txt'
To run the application type python app.py

--------

Datasets
--------
All sound files are splitted for many files by 25-milliseconds.
Then, every file is converted into spectrogram, which are saved to pitchdf. Then, it runs through the trained model and tune pitchs for the right pitches, mentioned in flutes_notes_frequencies dataframe.


* pitchdf -           dataframe with spectrogram of every audio- file and with the name of this note + octave (for example: 'A#1').
* notes_frequency -   dataframe with names of notes + octaves (for ex. 'A#1') and it's frequency in Hz
* origin_note_table - dataframe with names of notes (for ex. 'A#') and it's octave (1,2,3 и т.д.)
* TinySOL_metadata -  original dataframe TinySOL with file paths, names of notes etc.
* flutes_notes_frequencies - dataframe with actual notes frequencies for flute


Model
------------

Model used in this profect is a CNN.
The brief spectrograms from center of the audio sample will be used to train the CNN to predict pitch.

Please note
------------
* Symphonic Masks only supports MP4 and MOV videos
* Model trained on 1-channel audio, so if the input video has more channels, it will be first resampled to 1-channel audio.
