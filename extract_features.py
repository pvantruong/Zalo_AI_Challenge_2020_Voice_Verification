import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from numpy import savetxt
from numpy import loadtxt
import csv




#########
#train_datafram
filelist = os.listdir('all_files')
file_df = pd.DataFrame(filelist)
file_df = file_df.rename(columns={0:'file'})
speaker = []
for i in range(0, len(file_df)):
    if 'speaker' in file_df['file'][i]:
        speaker.append(227) #227 is the only numer has this feature
    elif '-' in file_df['file'][i]: 
        speaker.append(file_df['file'][i].split('-')[0])
    else:
        speaker.append(file_df['file'][i].split('.')[0])
file_df['speaker'] = speaker



#Feature Extraction Function:
def extract_features(files):
    print("Dang extract feature from: " + str(files.file))
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(os.path.abspath('all_files')+'/'+str(files.file))
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz



file_features = file_df.apply(extract_features, axis=1)


features_file = []
for i in range(0, len(file_features)):
    features_file.append(np.concatenate((
        file_features[i][0],
        file_features[i][1], 
        file_features[i][2], 
        file_features[i][3],
        file_features[i][4]), axis=0))

data_extracted = np.array(features_file)

savetxt('extracted_data.csv', data_extracted, delimiter=',')
###############################################



