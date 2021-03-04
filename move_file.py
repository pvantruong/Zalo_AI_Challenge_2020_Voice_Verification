import shutil
import pandas as pd 
import os


folder_list = os.listdir('dataset')
folder_df = pd.DataFrame(folder_list)
folder_df = folder_df.rename(columns={0:'folder'})
print(folder_df)

for i in range(0, len(folder_df)):
    file_list = os.listdir('./dataset/' + str(folder_df['folder'][i]))
    file_list = pd.DataFrame(file_list)
    file_list = file_list.rename(columns={0:'file'})
    for j in range(0, len(file_list)):
        shutil.copyfile('./dataset/'+ str(folder_df['folder'][i]) + '/' + str(file_list['file'][j]), './all_files/' + str(file_list['file'][j]))
