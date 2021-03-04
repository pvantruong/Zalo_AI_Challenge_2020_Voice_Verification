import os
import pandas as pd
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import csv
import ast
#import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
#######
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

###################
# filelist = os.listdir('all_files')
# file_df = pd.DataFrame(filelist)
# file_df = file_df.rename(columns={0:'file'})
# speaker = []
# for i in range(0, len(file_df)):
#     if 'speaker' in file_df['file'][i]:
#         speaker.append(227) #227 is the only numer has this feature
#     elif '-' in file_df['file'][i]:
#         speaker.append(file_df['file'][i].split('-')[0])
#     else:
#         speaker.append(file_df['file'][i].split('.')[0])
# file_df['speaker'] = speaker

##########
#extracted_data = loadtxt('extracted_data.csv', delimiter=',')
#shape: (10559, 193)
pair_list = pd.read_csv('pair_lists_new_little.csv') #, converters={'data_1': from_np_array, 'data_2': from_np_array})

# #####
# X_train = []
# for i in range(0, 160000):
#     X_train.append(np.concatenate((pair_list['data_1'][i], pair_list['data_2'][i])))
# X_train = np.array(X_train)

# ####
# print('Xong X_train')
# X_vali = []
# for i in range(160000, 190000):
#     X_vali.append(np.concatenate((pair_list['data_1'][i], pair_list['data_2'][i])))
# X_vali = np.array(X_vali)
# ####
# # X_test = []
# # for i in range(180000, 190000):
# #     X_test.append(np.concatenate((pair_list['data_1'][i], pair_list['data_2'][i])))
# # X_test = np.array(X_test)
# print('Xong X_test')

######## Save data into file ###
# savetxt('arrar_x_train_data.csv', X_train, delimiter=',')
# savetxt('array_x_vali_data.csv', X_vali, delimiter=',')
#X_vali = extracted_data[8000:10000]
#X_test = extracted_data[10000:]
#######

X_train = loadtxt('new_array_x_train_data.csv', delimiter=',')
X_vali = loadtxt('new_array_x_vali_data.csv', delimiter=',')
X_test = loadtxt('new_array_x_test_data.csv', delimiter=',')
##########
y_train = np.array(pair_list['label'][:240000])
y_vali = np.array(pair_list['label'][240000:280000])
y_test = np.array(pair_list['label'][280000:290000])
##########
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_vali = to_categorical(lb.fit_transform(y_vali))
##########
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_vali)
X_test = ss.transform(X_test)

############# MODEL #############256 512
print('Train model nha : D')
#Build Model NN:
# Build a simple dense model with early stopping and softmax for categorical classification, remember we have 30 classes
model = Sequential() #private_6
model.add(Dense(386, input_shape=(386,), activation = 'relu'))
model.add(Dropout(0.8))
model.add(Dense(32, activation = 'relu'))  #58.9
model.add(Dropout(0.7))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
#Pass data into model:
history = model.fit(X_train, y_train, batch_size=256, epochs=100, 
                    validation_data=(X_val, y_vali),
                    callbacks=[early_stop])
#pri_03: 09317
#pri_04: 0.922
#pri_05: 0.9261
#pri_06: 09351
#### PLOT GRAPH #######
# Check out our train accuracy and validation accuracy over epochs.
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
# Set figure size.
plt.figure(figsize=(12, 8))
# Generate line plot of training, testing loss over epochs.
plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
# Set title
plt.title('private_6', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(0,100,5), range(0,100,5))
plt.legend(fontsize = 18)
plt.show()




##################################################
######## Test ###########
# X_test_1 = loadtxt('Zalo_feautures_data_1.csv', delimiter=',')
# print('Load xong X_test_1')
# X_test_2 = loadtxt('Zalo_feautures_data_2.csv', delimiter=',')
# print('Load xong X_test_2')
# ####
# X_Zalo_test = []
# for i in range(0, len(X_test_1)):
#     X_Zalo_test.append(np.concatenate((X_test_1[i], X_test_2[i])))
# X_Zalo_test= np.array(X_Zalo_test)
# ###
# X_Zalo_test = ss.transform(X_Zalo_test)



# predictions = model.predict_classes(X_Zalo_test)
# #We transform back our predictions to the speakers ids
# predictions = lb.inverse_transform(predictions)
# #Finally, we can add those predictions to our original dataframe
# Zalo_public_test = pd.read_csv('public-test.csv')
# Zalo_public_test['label'] = predictions
# ###
# Zalo_public_test.to_csv('OUTPUT_17.csv')

# model.save('model_11')

model.save('model_18')

# We get our predictions from the test data
predictions = model.predict_classes(X_test)
# We transform back our predictions to the speakers ids
predictions = lb.inverse_transform(predictions)
print(predictions)
answer_df = pd.DataFrame()
answer_df['label'] = y_test
answer_df['prediction'] = predictions
count = 0
for i in range(0, len(answer_df['label'])):
    if answer_df['label'][i] == answer_df['prediction'][i]:
        count += 1
print('Count: ', count)
print('Len: ', len(answer_df['label']))
print('Rate: ', count / len(answer_df['label']))
# Finally, we can add those predictions to our original dataframe
# pair_list['predictions'] = predictions

# ########
# # Code to see which values we got wrong
# pair_list[pair_list['label'] != pair_list['predictions']]
# # Code to see the numerical accuracy
# (1-round(len(pair_list[pair_list['speaker'] != pair_list['predictions']])/len(pair_list),3))*100


