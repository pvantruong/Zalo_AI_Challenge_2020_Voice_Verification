#from pydub import AudioSegment
import os
import pandas as pd
import numpy as np
import random
from numpy import savetxt
from numpy import loadtxt
import csv

##########
extracted_data = loadtxt('extracted_data.csv', delimiter=',') 

filelist = os.listdir('all_files')
file_df = pd.DataFrame(filelist)
file_df = file_df.rename(columns={0:'file'})
###
columns = ['audio_1', 'audio_2']
csv_df = pd.DataFrame(columns=columns)
audio_1 = []
audio_2 = []
data_1 = []
data_2 = []
##########
def check_label(i, j):
    
    if 'speaker' in file_df['file'][i]:
        speaker_1 = 272
    elif '-' in file_df['file'][i]: 
        speaker_1 = file_df['file'][i].split('-')[0]
    else:
        speaker_1 = file_df['file'][i].split('.')[0]

    if 'speaker' in file_df['file'][j]:
        speaker_2 = 272
    elif '-' in file_df['file'][j]: 
        speaker_2 = file_df['file'][j].split('-')[0]
    else:
        speaker_2 = file_df['file'][j].split('.')[0]
    if speaker_1 == speaker_2:
        return 1
    else:
        return 0
##########################
for i in range(0, len(file_df)):
    print(i)
    for j in range(1, 40):
        if i + j >= len(file_df):
            break
        #temp = pd.DataFrame({'audio_1': [file_df['file'][i]], 'audio_2':[file_df['file'][i+j]]})
        #csv_df.append(temp)
        #csv_df['audio_1'][count] = file_df['file'][i]
        #csv_df['audio_2'][count] = file_df['file'][i+j]
        if check_label(i, i+j) == 1:
            audio_1.append(file_df['file'][i])
            audio_2.append(file_df['file'][i+j])
            data_1.append(extracted_data[i])
            data_2.append(extracted_data[i + j])
    for j in range(1, 15): #
        if i < 32:
            random_range = range(i+30, len(file_df))
        elif i > len(file_df) - 32:
            random_range = range(0, i - 30)
        else:
            random_range = [*range(0, i - 30), *range(i+30, int(len(file_df)))]
        temp = random.choice(random_range)
        #csv_df['audio_1'][count] = file_df['file'][i]
        #csv_df['audio_2'][count] = file_df['file'][random.choice(random_range)]
        #temp = pd.DataFrame({'audio_1': [file_df['file'][i]], 'audio_2':[file_df['file'][random.choice(random_range)]]})
        #csv_df.append(temp)
        audio_1.append(file_df['file'][i])
        audio_2.append(file_df['file'][temp])
        data_1.append(extracted_data[i])
        data_2.append(extracted_data[temp])
print('Xong Pair file')
csv_df['audio_1'] = audio_1
csv_df['audio_2'] = audio_2
csv_df['data_1'] = data_1
csv_df['data_2'] = data_2
#print(type(csv_df['data_1'][1]))
#### Set speaker #######
speaker_1 = []
for i in range(0, len(csv_df)):
    if 'speaker' in csv_df['audio_1'][i]:
        speaker_1.append(272) #227 is the only numer has this feature
    elif '-' in csv_df['audio_1'][i]: 
        speaker_1.append(csv_df['audio_1'][i].split('-')[0])
    else:
        speaker_1.append(csv_df['audio_1'][i].split('.')[0])
csv_df['speaker_1'] = speaker_1
#####
speaker_2 = []
for i in range(0, len(csv_df)):
    if 'speaker' in csv_df['audio_2'][i]:
        speaker_2.append(272) #227 is the only numer has this feature
    elif '-' in csv_df['audio_2'][i]: 
        speaker_2.append(csv_df['audio_2'][i].split('-')[0])
    else:
        speaker_2.append(csv_df['audio_2'][i].split('.')[0])
csv_df['speaker_2'] = speaker_2
print(csv_df)
### label ###
print('Di label ne')
label = []
for i in range(0, len(csv_df)):
    if csv_df['speaker_1'][i] == csv_df['speaker_2'][i]:
        label.append(1)
    else:
        label.append(0)
csv_df['label'] = label
csv_df.to_csv('pair_lists_new.csv')


# for i in range(0, len(train_df)):
#     a = random.randrange(0,len(train_df))
#     b = random.randrange(0,len(train_df))
    

# df.to_csv(index=False)