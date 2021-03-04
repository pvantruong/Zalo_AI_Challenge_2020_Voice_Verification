import os
import pandas as pd
import numpy as np
import librosa
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from keras.utils.np_utils import to_categorical
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.callbacks import EarlyStopping
# from keras.models import model_from_json
from numpy import savetxt
from numpy import loadtxt
import csv






#test_dataframe
filelist = os.listdir('private-test')
test_df = pd.DataFrame(filelist)
test_df = test_df.rename(columns={0:'file'})
Zalo_list_file_df = test_df

#print(test_df)
####### Zalo test set #########
Zalo_test_link = pd.read_csv('private-test.csv')


#Feature Extraction Function:
def extract_features(files):
    print("Dang extract feature from: " + str(files.file))
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(os.path.abspath('private-test')+'/'+str(files.file))
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


####################################    
#Use for the first time run code
#After that use the load file below: 
#Extract:
# train_features = train_df.apply(extract_features, axis=1)
# vali_features = vali_df.apply(extract_features, axis=1)
#test_features = test_df.apply(extract_features, axis=1)


##### Zalo ##################
Zalo_file_features = test_df.apply(extract_features, axis=1)

print('Extract xong Zalo file')
#Mapping features to list_link_file_test:
print('Now move to mapping: ') 
for i in range(0, len(Zalo_test_link['audio_1'])):
    j = 0
    for j in range(0, len(Zalo_list_file_df)):
        if Zalo_test_link['audio_1'][i] == Zalo_list_file_df['file'][j]:
            break
    Zalo_test_link['audio_1'][i] = Zalo_file_features[j]
    print(i)
##
print('Dang mapping audio 2')
for i in range(0, len(Zalo_test_link['audio_2'])):
    j = 0
    for j in range(0, len(Zalo_list_file_df)):
        if Zalo_test_link['audio_2'][i] == Zalo_list_file_df['file'][j]:
            break
    Zalo_test_link['audio_2'][i] = Zalo_file_features[j]
print('Mapping xong xuoi')
#############
Zalo_features_test_1 = []
Zalo_features_test_2 = []
for i in range(0, len(Zalo_test_link['audio_2'])):
    Zalo_features_test_1.append(np.concatenate((
        Zalo_test_link['audio_1'][i][0],
        Zalo_test_link['audio_1'][i][1], 
        Zalo_test_link['audio_1'][i][2], 
        Zalo_test_link['audio_1'][i][3],
        Zalo_test_link['audio_1'][i][4]), axis=0))
    Zalo_features_test_2.append(np.concatenate((
        Zalo_test_link['audio_2'][i][0],
        Zalo_test_link['audio_2'][i][1], 
        Zalo_test_link['audio_2'][i][2], 
        Zalo_test_link['audio_2'][i][3],
        Zalo_test_link['audio_2'][i][4]), axis=0))

X_test_1 = np.array(Zalo_features_test_1)
X_test_2 = np.array(Zalo_features_test_2)
########### End Zalo #######################
#### Save for the first time ###########
savetxt('Zalo_private_data_1.csv', X_test_1, delimiter=',')
savetxt('Zalo_private_data_2.csv', X_test_2, delimiter=',')
