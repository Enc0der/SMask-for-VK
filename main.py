
import pandas as pd
import sklearn
import IPython.display as ipd
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import cross_val_score
import random
from math import *
import librosa.display
import pylab as pl
# import copy
# import os
import scipy.fftpack
import sounddevice as sd
import time
import scipy.signal
from PIL import Image
from pathlib import Path
from pylab import rcParams
rcParams['figure.figsize'] = 14, 6
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
#Reports
from sklearn.metrics import classification_report, confusion_matrix
import sounddevice as sd
from scipy.io.wavfile import write
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf


from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from IPython.display import HTML
style = '<style>svg{width: 50% !important; height: 50% !important;}<style>'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import sounddevice as sd
from scipy.io.wavfile import write
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
import soundfile as sf

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from IPython.display import HTML

style = '<style>svg{width: 50% !important; height: 50% !important;}<style>'

#  Записываем звук в рилз:

import sounddevice as sd
import soundfile as sf

fs = 44100
seconds = 30
scale_file = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
sf.write('train_file.wav', scale_file, fs)

# загружаем файл в нейронку для создания учебной таблицы:

import crepe
from scipy.io import wavfile

sr, audio = wavfile.read('/Users/monglels/Desktop/Обучение SymphonicMasks/train_file.wav')
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

# также загружаем тестовый файл:


ee_train = time, frequency, confidence, activation

df = pd.DataFrame(ee_train)
df.to_csv('file2.csv', index=True
          , header=True)

new = df.T

new.columns = ['time', 'frequency', 'confidence', 'new_col4']

new = new.drop(columns={'new_col4'})

new.to_csv(r'/Users/monglels/flute_input.csv')

melody_df_train = pd.read_csv('/Users/monglels/Desktop/flute_input.csv', delimiter=";")
melody_df_train = melody_df_train.fillna(0)
melody_df_train = melody_df_train.drop(columns={'Unnamed: 0'})
melody_df_train.rename(columns={'Note or not ': 'Note_or_not'}, inplace=True)
melody_df_train

# количество нот
len(melody_df_train[melody_df_train['Note or not '] == 1])

y, sr = librosa.load('/Users/monglels/Desktop/Обучение SymphonicMasks/train_file.wav')
print(type(y), type(sr))

# воспроизводим звук без обработки
import IPython.display as ipd

plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
ipd.Audio('/Users/monglels/Desktop/Обучение SymphonicMasks/train_file.wav')

# Загружаем файл и начинаем обучение модели по нахождению нот:


X_train = melody_df_train.drop(['Note_or_not'], axis=1)
y_train = melody_df_train.Note_or_not
X_test = melody_df_test.drop(['Note_or_not'], axis=1)
y_test = melody_df_test.Note_or_not

from sklearn.model_selection import GridSearchCV

parameters = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}

grid_search_cv_clf = GridSearchCV(clf, parameters, cv=5)

grid_search_cv_clf.fit(X_train, y_train)

grid_search_cv_clf.best_params_

best_clf = grid_search_cv_clf.best_estimator_

best_clf.score(X_test, y_test)

best_clf.fit(X_test, y_test)

y_pred = best_clf.predict(X_test)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred)

y_predicted_prob = best_clf.predict_proba(X_test)

y_pred = np.where(y_predicted_prob[:, 1] > 0.8, 1, 0)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred)


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier()

parametrs = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}

grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=5)

grid_search_cv_clf.fit(X_train, y_train)

best_classifier = grid_search_cv_clf.best_estimator_

feature_importances = best_clf.feature_importances_

feature_importances_df = pd.DataFrame({'features': list(X_train),
                                       'feature_importances': feature_importances})

feature_importances_df.sort_values('feature_importances', ascending=False)

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, recall_score

best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10
                                       )

cross_val_score(clf, X_test, y_test, cv=5).mean()

# Вырезаем отдельные файлы по таймингу датафрейма:

import wave

for i in range(len(melody_with_differences)):
    # times between which to extract the wave from
    start = starts[i]  # seconds
    end = ends[i]  # seconds

    # file to extract the snippet from
    with wave.open('input_file.wav', "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        infile.setpos(int(start * framerate))
        # extract data
        data = infile.readframes(int((end - start) * framerate))

    # write the extracted data to a new file
    with wave.open('my_out_file.wav', 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)

# ## Создаем переменные со значениями частоты всех нот флейты:

# In[73]:


h = 248.96
c1 = 263.25
c1_sharp = 279.18
d1 = 295.66
d1_sharp = 313.13
e1 = 331.13
f1 = 351.23
f1_sharp = 371.99
g1 = 394.00
g1_sharp = 417.30
a1 = 442.00
a1_sharp = 462.16
h1 = 495.66
c2 = 525.25
c1_sharp = 556.36
d2 = 589.32
d2_sharp = 624.26
e2 = 661.26
f2 = 698.46
f2_sharp = 741.96
g2 = 786.00
g2_sharp = 832.60
a2 = 882.00
a1_sharp = 934.32
h2 = 969.75
c3 = 1048.30
c3_sharp = 1108.70
d3 = 1176.60
d3_sharp = 1246.50
e3 = 1318.50
f3 = 1398.90
f2_sharp = 1482.00
g3 = 1568.00
g2_sharp = 1663.20
a3 = 1722.00
a2_sharp = 1848.60
h3 = 1977.50
c4 = 2095.00
c4_sharp = 2219.40
d4 = 2351.20

# ## Сравниваем частоты из таблицы с данными переменными:

# In[60]:


# Вырезаем ноты по отдельности
import wave

for i in range(len(a)):
    # times between which to extract the wave from
    start = 1  # seconds
    end = 3  # seconds

    # file to extract the snippet from
    with wave.open('input_file.wav', "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        infile.setpos(int(start * framerate))
        # extract data
        data = infile.readframes(int((end - start) * framerate))

    # write the extracted data to a new file
    with wave.open('my_out_file.wav', 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)

# In[61]:


ipd.Audio('my_out_file.wav')



# clf = tree.DecisionTreeClassifier(criterion='entropy')

# clf.fit(X, y)
#
# plt.figure(figsize=(100, 25))
# tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# clf.fit(X_train, y_train)
#
# clf.fit(X_test, y_test)
#
# max_depth_values = range(1, 100)
#
# scores_data = pd.DataFrame()
#
# for max_depth in max_depth_values:
#     clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
#     clf.fit(X_train, y_train)
#     train_score = clf.score(X_train, y_train)
#     test_score = clf.score(X_test, y_test)
#
#     mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
#     temp_score_data = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score],
#                                     'test_score': [test_score], 'cross_val_score': [mean_cross_val_score]})
#     scores_data = pd.concat([scores_data, temp_score_data])


# In[207]:


# scores_data_long = pd.melt(scores_data, id_vars = ['max_depth'],
#                 value_vars = ['train_score','test_score',  'cross_val_score' ],
#                 var_name = 'set_type', value_name = 'score')
#
# sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
#
#
# # In[208]:
#
#
# scores_data_long.query('set_type == "cross_val_score"').head(20)
#
#
# plt.figure(figsize=(100, 25))
# tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)


from sklearn.model_selection import GridSearchCV

parameters = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}

grid_search_cv_clf = GridSearchCV(clf, parameters, cv=5)

grid_search_cv_clf.fit(X_train, y_train)

grid_search_cv_clf.best_params_

best_clf = grid_search_cv_clf.best_estimator_

best_clf.score(X_test, y_test)

best_clf.fit(X_test, y_test)

y_pred = best_clf.predict(X_test)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred)

y_predicted_prob = best_clf.predict_proba(X_test)

pd.Series(y_predicted_prob[:, 1]).hist()


pd.Series(y_predicted_prob[:, 1]).unique()


y_pred = np.where(y_predicted_prob[:, 1] > 0.8, 1, 0)


precision_score(y_test, y_pred)


recall_score(y_test, y_pred)


from sklearn import metrics


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predicted_prob[:, 1])
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier()

parametrs = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}

grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=5)

grid_search_cv_clf.fit(X_train, y_train)

best_classifier = grid_search_cv_clf.best_estimator_

feature_importances = best_clf.feature_importances_

feature_importances_df = pd.DataFrame({'features': list(X_train),
                                       'feature_importances': feature_importances})

feature_importances_df.sort_values('feature_importances', ascending=False)

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, recall_score

best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10
                                       )

cross_val_score(clf, X_test, y_test, cv=5).mean()

# Вырезаем отдельные файлы по таймингу датафрейма:


# Вырезаем ноты по отдельности
import wave

# массив из time где начинается новая частота
starts = []
# массив из time где заканчивается новая частота
ends = []
for i in range(len(melody_with_differences)):
    # times between which to extract the wave from
    start = starts[i]  # seconds
    end = ends[i]  # seconds

    # file to extract the snippet from
    with wave.open('input_file.wav', "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        infile.setpos(int(start * framerate))
        # extract data
        data = infile.readframes(int((end - start) * framerate))

    # write the extracted data to a new file
    with wave.open('my_out_file.wav', 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)

# ## Создаем переменные со значениями частоты всех нот флейты:

# In[73]:


h = 248.96
c1 = 263.25
c1_sharp = 279.18
d1 = 295.66
d1_sharp = 313.13
e1 = 331.13
f1 = 351.23
f1_sharp = 371.99
g1 = 394.00
g1_sharp = 417.30
a1 = 442.00
a1_sharp = 462.16
h1 = 495.66
c2 = 525.25
c1_sharp = 556.36
d2 = 589.32
d2_sharp = 624.26
e2 = 661.26
f2 = 698.46
f2_sharp = 741.96
g2 = 786.00
g2_sharp = 832.60
a2 = 882.00
a1_sharp = 934.32
h2 = 969.75
c3 = 1048.30
c3_sharp = 1108.70
d3 = 1176.60
d3_sharp = 1246.50
e3 = 1318.50
f3 = 1398.90
f2_sharp = 1482.00
g3 = 1568.00
g2_sharp = 1663.20
a3 = 1722.00
a2_sharp = 1848.60
h3 = 1977.50
c4 = 2095.00
c4_sharp = 2219.40
d4 = 2351.20

# ## Сравниваем частоты из таблицы с данными переменными:

# In[60]:


# Вырезаем ноты по отдельности
import wave

for i in range(len(a)):
    # times between which to extract the wave from
    start = 1  # seconds
    end = 3  # seconds

    # file to extract the snippet from
    with wave.open('input_file.wav', "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        infile.setpos(int(start * framerate))
        # extract data
        data = infile.readframes(int((end - start) * framerate))

    # write the extracted data to a new file
    with wave.open('my_out_file.wav', 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)

# In[61]:


ipd.Audio('my_out_file.wav')

