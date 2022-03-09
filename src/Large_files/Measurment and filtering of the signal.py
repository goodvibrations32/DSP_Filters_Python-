
# %%
#Use of pandas library for reading hdf5 file format

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

#%%
#Read and store the .h5 file with pandas
f_1 = pd.HDFStore(path='/run/media/goodvibrations/KINGSTON/_noiseReference_2021/20210831/noise_reference_raw.h5', mode='r')
#print(f_1.keys())
#print(f_1.get('/df'))
data_raw = f_1['/df']
print(data_raw.tail())

first_column =data_raw[['raw-2021.08.31-12:26:54']]

time = np.linspace(0, 7.599998, 3800000)
sig = np.array(first_column)

plt.plot(time, sig)
plt.show()

#print (first_column.head())

#%%
# walk () function for reading .h5 files
for (path, subgroups, subkeys) in f_1.walk():
    for subgroup in subgroups:
        print ("GROUP:{}/{}".format(path, subgroup))
    for subkey in subkeys:
        key= "/".join([path, subkey])
        print ("KEY:{}".format(key))
        print (f_1.get(key))

#print(data_raw.info())

#print(list(data_raw.keys()))

#print(data_raw)
#first_column =data_raw[['raw-2021.08.31-12:26:54']] 
#print (first_column.head())
#print(list(first_column.keys()))

#print (np.array(first_column))
#print(first_column)
#x = np.linspace(0, 8, 3800000, True)

#plt.plot(x,first_column)


#%%
from scipy import signal

# FIR Low-pass filter 
fs = 500000        #sampling freq
cutoff = 150     #Cutoff freq for gain drop
trans_width = 80 # width of transition from passband to stopband in Hz
numtaps = 250     #Size of the FIR filter

#Construct a Low-pass FIR filter 
taps = signal.remez(numtaps, [0, cutoff, cutoff+trans_width, 0.5*fs], [1,0], Hz= fs)
w, h = signal.freqz(taps, [1], worN=2000)

#Applying the FIR LP filter to the signal 
y = signal.lfilter(taps, 1.0, sig)
fig, (ax1, ax2) = plt.subplots( 2, 1, sharex=False)
ax1.plot(sig)
ax1.set_title('10 and 20 Hz sinusoids with 8kHz interferences')
#ax1.axis([0, 1, -15, 15])
ax2.plot (y)
ax2.set_title('After Filtering the 8kHz with FIR filter (signal.remez)')
#ax2.axis([0, 1, -5, 5])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()

# %%
# Fast Fourier Transformation of input and output signals 
f0=3000
N = int(10*(fs/f0))   

yf_input = np.fft.rfft(sig) 
y_input_mag_plot = np.abs(yf_input)/N


f= np.linspace (0, (N-1)*(fs/N),N )

f_plot = f[0:int(N/2+1)]
y_input_mag_plot = 2*y_input_mag_plot[0:int(N/2+1)]
y_input_mag_plot[0] = y_input_mag_plot[0] / 2

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
fig.suptitle('IIR butterworth Low-pass Filter')
ax1.plot(f_plot, y_input_mag_plot)
ax1.grid(alpha=0.3)

ax1.set_ylabel('Amplitute [dB]')

yf_output = np.fft.rfft(y)
y_output_mag_plot = np.abs(yf_output)/N
y_output_mag_plot = 2* y_output_mag_plot[0:int(N/2+1)]
y_output_mag_plot[0]= y_output_mag_plot[0]/2  
ax2.plot(f_plot, y_output_mag_plot)

plt.plot(f_plot, y_output_mag_plot, label='output signal')
plt.grid(alpha=0.3)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitute [dB]')
#plt.xscale('log')
plt.legend()

plt.show()

#%%
#Plot the signal in frequency domain after using FIR LP (signal.remez) 
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
fig.suptitle('FIR Low-pass Filter (signal.remez)')
ax1.plot(f_plot, y_input_mag_plot, label= 'input signal')
ax1.grid(alpha=0.3)

ax1.set_ylabel('Amplitute [dB]')

yf_output = np.fft.rfft(y)
y_output_mag_plot = np.abs(yf_output)/N
y_output_mag_plot = 2* y_output_mag_plot[0:int(N/2+1)]
y_output_mag_plot[0]= y_output_mag_plot[0]/2  
ax2.plot(f_plot, y_output_mag_plot)

plt.plot(f_plot, y_output_mag_plot, label= 'output signal')
plt.grid(alpha=0.3)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitute [dB]')
plt.xscale('log')
plt.legend()

plt.show()

# %%
# Power Spectrum / signal.welch

sig=np.array(first_column)
f, Pxx_spec = signal.welch(sig, fs, window ='flattop', nperseg = 1, scaling = 'density')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))

#plt.xlim(0, 8)

plt.xlabel('Frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum before filter is applied')
plt.show()

# %%
#FIR filtered signal Power Spectrum 
fs = 44000
f, Pxx_spec = signal.welch(y, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlim(0,10000)
plt.xlabel ('Frequency [Hz]')
plt.ylabel ('Linear Spectrum [V RMS]')
plt.title (' FIR LP filtered signal Power Spectrum')
plt.show ()





#%%
f_2 = pd.HDFStore(path='/home/goodvibrationskde/Documents/noise_reference_filt.h5', mode='r')
print(f_2.keys())
#print(f_1.get('/df'))
data_filt = f_2['/df']
first_column =data_filt[['filt-2021.08.31-12:26:54']] 
print(first_column)


#print(data)
#print(data)
#print(max(data))
#data.head
#print(f_1.groups())



#print(list(f_1.keys()))

#print(f_1.get('/df'))
#print (min(f_1))

f_1.close()

# %%
