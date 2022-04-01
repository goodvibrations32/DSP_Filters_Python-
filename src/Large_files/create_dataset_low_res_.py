#%%

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

#%%
file_path = input('The full path of raw data file to process: ' )
file_name_of_raw =input('Enter the name of the raw signal file :') 
#%%
#Read and store the .h5 file with pandas
f_1 = pd.HDFStore(path=f'{file_path}{file_name_of_raw}', mode='r')

print('The data frame key is: ',f_1.keys())

data_fr_key = input('Input the data frame key of raw signal file:' )

data_raw = f_1[data_fr_key]
print(data_raw.info())

L = list(data_raw.keys())

print (L)


#%%
#MAKE IT BETTER
raw_sig_df =[]

for element1 in L:
    raw_sig_df.append(data_raw.get(L))

sig_array= []
for element2 in L:
    sig_array.append(np.array(raw_sig_df))

#%%
#////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////
#The only filter with les standard deviation on output from input
#Somehow should test more of these kind of filters with no window functions 

#FIR Low-pass filter on the signal with the inverter connected and off

#The length of the filter(number of coefficients, the filter order + 1)
numtaps_2 = 2000
fs = 500000
cutoff_hz = 0.00001
nyq_rate = fs/2 

#Use of firwin to create low-pass FIR filter
fir_co = signal.firwin(numtaps_2, cutoff_hz)
w_fir_co, h_fir_co = signal.freqz(fir_co, [1])

#%%
# LET'S MAKE IT BETTER
blank_test =[]
experiment = []
for element3 in sig_array:
    blank_test = signal.lfilter(fir_co, 1.0, element3)
    experiment.append(blank_test)

#%%

#=====================================================
#++++++++ Plot original and filtered signal+++++++++++ 
#=====================================================
#%%
#Time interval of the samples
time = np.linspace(0, 7.599998, len(data_raw))

#The first N-1 samples are corrupted by the initial conditions
warmup = numtaps_2 - 1

#The phase delay of the filtered signal
delay= (warmup / 2) / fs

time_no_shift = time[warmup:]-delay


#%%
# signal shift for rejecting the corrupted signal from the 
#blank output of the filter


filt_data = []
for element4 in experiment:
    filt_data.append(element4[warmup:])


#%%
#Construct data frame better 
#using pandas library

#Rename old attributes to mach with filtered data
#signal

F = []
for i in range(0,len(L)):
    F.append(L[i].replace("raw","filt"))
i=0
#%%
#Create the new file
#Choose the desired directory to create the new file 
#and add the __NAME__.h5 at the end of the screen 
# WARNING : If file already exists in the dir and it is closed via :
#hf_st_pd_.close() it will be overwritten
#final_data = [filt_con_off, filt_con_on, filt_con_on_WS_5, filt_discon_off,filt_discon_on,filt_discon_on_WS_5]

#%%
new_file_name = input("""Enter the name of the new folder : 
the file will be created in the same path with the raw data file
if 0 is passed for a new name the old name is used with replacing
the "raw" at the end of the name with "filt"
""")

if new_file_name == '0' :
    file_name = file_name_of_raw.replace("raw", "filt")
else:
    file_name = new_file_name

hf_st_pd_ = pd.HDFStore(f'{file_path}{file_name}', mode='w')

df2 = pd.DataFrame({
        
    F[0]:filt_data[0],
    F[1]:filt_data[1],
    F[2]:filt_data[2],
    F[3]:filt_data[3],
    F[4]:filt_data[4],
    F[5]:filt_data[5]
    }
,index=(time_no_shift)
)

hf_st_pd_.put('df_filt', df2, format='table', data_columns=True)
    
hf_st_pd_.close()


#%%
#Read the file that was just created

f_3 = pd.HDFStore(path=f'{file_path}{file_name}',mode='r')

data_filt = f_3['df_filt']
# %%
#This is added in order to avoid manual close of the filtered data file
f_3.close()


