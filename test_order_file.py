import os
import pandas as pd
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import csv
import ast
import librosa

#train_datafram
filelist = os.listdir('train_set')
train_df = pd.DataFrame(filelist)
train_df = train_df.rename(columns={0:'file'})
speaker = []
for i in range(0, len(train_df)):
    speaker.append(train_df['file'][i].split('-')[0])
train_df['speaker'] = speaker
#print(train_df)


#Feature Extraction Function:
def extract_features(files):
    print("Dang extract feature from: " + str(files.file))
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(os.path.abspath('Zalo_test_set')+'/'+str(files.file))
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



train_features = train_df.apply(extract_features, axis=1)

#Concatenate into single array:
features_train = []
for i in range(0, len(train_features)):
    features_train.append(np.concatenate((
        train_features[i][0],
        train_features[i][1], 
        train_features[i][2], 
        train_features[i][3],
        train_features[i][4]), axis=0))
####

X_train = np.array(features_train)
