#%%

import numpy as np
import pandas as pd
from scipy import signal
from Plots_time_freq_domain_ import UNCORR, TIME_NO_SHIFT, L, file_name_of_raw, file_path

#%%
#Construct data frame better 
#using pandas library

#Rename old attributes to mach with filtered data
#signal

F = []
for i in range(0,len(L)):
    F.append(L[i].replace("raw","HD"))
i=0

#Create a dictionary from the lists of modified keys and filtered output 
a = dict(zip(F, UNCORR))

#Create a pandas dataframe for the generated file 
df=pd.DataFrame(data=a, index=None)
df.insert(loc=0, column="Time", value=TIME_NO_SHIFT)

#%%
#Create the new file
#Choose the desired directory to create the new file 
#and add the __NAME__.h5 at the end of the screen 
# WARNING : If file already exists in the dir and it is closed via :
#hf_st_pd_.close() it will be overwritten

new_file_name = input("""Enter the name of the new file : 
the file will be created in the same path folder with the raw data 
if 0 is passed for a new name the old name is used with replacing
the "raw" at the start of the name with "HD"
""")

if new_file_name == '0' :
    file_name = file_name_of_raw.replace("raw", "HD")
else:
    file_name = new_file_name

#HDF5 file in same path with raw signal folder diff name
hf_st_pd_ = pd.HDFStore(f'{file_path}{file_name}', mode='w')
hf_st_pd_.put('df_filt', df, format='table', data_columns=True)
hf_st_pd_.close()
#%%
#Read the file that was just created

f_3 = pd.HDFStore(path=f'{file_path}{file_name}',mode='r')

data_filt = f_3['df_filt']
# %%
#This is added in order to avoid manual close of the filtered data file
f_3.close()