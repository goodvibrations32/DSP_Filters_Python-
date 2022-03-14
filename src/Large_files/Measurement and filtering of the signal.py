
# %%
#Use of pandas library for reading hdf5 file format

from markupsafe import t
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft

#%%
# Define function for Filter freq response

def plot_response(fs, w, h, title):
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    plt.ylim(-40, 5)
    #plt.xlim(0, 0.5*fs)
    plt.xscale('log')
   
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title(title)

#%%
#Read and store the .h5 file with pandas
f_1 = pd.HDFStore(path='/run/media/goodvibrations/KINGSTON/_noiseReference_2021/20210831/noise_reference_raw.h5', mode='r')
f_1_filt = pd.read_hdf('/run/media/goodvibrations/KINGSTON/_noiseReference_2021/20210831/noise_reference_filt.h5', mode='r')

#print (f_1_filt.info())
#print(f_1.keys())
#print (list(f_1_filt.keys()))
#print(f_1.get('/df'))
data_raw = f_1['/df']
data_filt = f_1_filt.get('filt-2021.08.31-12:26:54')
#print (data_filt.head())
print(data_raw.head())
#print (data_filt)
#print (data_filt.shape)

first_column =data_raw.get('raw-2021.08.31-12:26:54')
first_column.info()

#%%
#reading of list
sig= []
time = np.linspace(0, 7.599998, 3800000)
sig.append(first_column)

print(sig)


#print(np.array(sig))


#sig = np.array(first_column)

#plt.plot(time, sig)
#plt.show()

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
cutoff = 8960     #Cutoff freq for gain drop
trans_width = 80 # width of transition from passband to stopband in Hz
numtaps = 250     #Size of the FIR filter

#Construct a Low-pass FIR filter 
taps = signal.remez(numtaps, [0, cutoff, cutoff+trans_width, 0.5*fs], [1,0], Hz= fs)
w, h = signal.freqz(taps, [1], worN=2000)

#Applying the FIR LP filter to the signal 
y = signal.lfilter(taps, 1.0, sig)

plt.title('Unfiltered Signal')
plt.grid(True)
plt.xlabel('Time in seconds [s]')
plt.ylabel('Amplitute [dB]')
plt.plot(time, sig)
plt.show()

plt.title('Filtered signal')
plt.grid(True)
plt.xlabel('Time in seconds [s]')
plt.ylabel('Amplitute [dB]')
plt.plot(time, y)
plt.show()

#%%
#standard diviation of filtered and original signal

std_filt= np.std(y)
std_raw= np.std(sig)

print (std_raw)
print (std_filt)

#%%
#Construct a Low-pass FIR LP filter with Kaiser Window method
nyq_rate = fs/2.0
width = 10000/nyq_rate
ripple_dB = 10
N, beta = signal.kaiserord(ripple_dB, width)
cutoff_hz = 150

taps_2 = signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
w_2, h_2 = signal.freqz(taps_2, [1], worN=2000)
y_2 = signal.lfilter(taps_2, 1.0, sig)


#%%
#FIR BP filter
#Construct a FIR BP filter with kaiser window method
#It seems that freq are somehow lost in the filtering process ??????
#I cannot take the same resoults out of Power spectrums and FFT plot in frequency domain

f1, f2= 100/nyq_rate, 1000/nyq_rate

taps_2_bandpass_kaiser = signal.firwin(numtaps, [f1, f2], pass_zero= False)
w_2_bp_kaiser,h_2_bp_kaiser = signal.freqz(taps_2_bandpass_kaiser, [1], worN=2000) 
y_2_bp_kaiser = signal.lfilter(taps_2_bandpass_kaiser, 1.0, sig)

#%%
#FIR bp filter for smooth starting point
f1, f2= 0.02/nyq_rate, 2/nyq_rate

taps_2_bandpass_clear = signal.firwin(numtaps, [f1, f2], pass_zero='bandpass')
w_2_bp_kaiser_clear,h_2_bp_kaiser_clear = signal.freqz(taps_2_bandpass_kaiser, [1], worN=2000) 

y_bp_base_clear= signal.lfilter(taps_2_bandpass_clear, 1.0, sig)

#%%
#Plot Frequency response of HP FIR filter  
plot_response(fs, w_2_bp_kaiser_clear, h_2_bp_kaiser_clear, "FIR BP Filter")
plot_response(fs, w_2_bp_kaiser, h_2_bp_kaiser, 'Band-pass for High freq ' )

#%%
#Plot the kaiser widnow method filter system response
plt.title('Unfiltered Signal')
plt.grid(True)
plt.xlabel('Time in seconds [s]')
plt.ylabel('Amplitute [dB]')
plt.plot(time, sig)
plt.show()

print(np.std(sig))

plt.title('Filtered signal BP Kaiser window method')
plt.grid(True)
plt.xlabel('Time in seconds window kaiser method [s]')
plt.ylabel('Amplitute [dB]')
plt.ylim([1.6, 1.75])
plt.plot(time, y_2_bp_kaiser)
plt.show()

print(np.std(y_2_bp_kaiser))

plt.title('BP FIR filter for canceling the bass freq 0-100 Hz')
plt.grid(True)
plt.xlabel('Time in seconds window kaiser method [s]')
plt.ylabel('Amplitute [dB]')
plt.ylim([1.58, 1.6])
plt.plot(time, y_bp_base_clear)
plt.show()

print(np.std(y_bp_base_clear))
#%%
plt.plot(data_filt)
plt.show()

# %%
# Power Spectrum / signal.welch

f, Pxx_spec = signal.welch(sig, fs, window ='flattop', nperseg = 1024, scaling = 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))

#plt.xlim(0, 8)

plt.xlabel('Frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum before filter is applied')
plt.show()

# %%
#FIR LP filtered signal Power Spectrum 

f, Pxx_spec = signal.welch(y_2_bp_kaiser, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
#plt.xlim(0,10000)
plt.xlabel ('Frequency [Hz]')
plt.ylabel ('Linear Spectrum [V RMS]')
plt.title (' FIR BP filter(kaiser window signal.firwin) filtered signal Power Spectrum')
plt.show ()

#%%
#FIR BP filtered signal Power Spectrum
f, Pxx_spec = signal.welch(y_bp_base_clear, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
#plt.xlim(0,10000)
plt.xlabel ('Frequency [Hz]')
plt.ylabel ('Linear Spectrum [V RMS]')
plt.title (' FIR BP filter first edge cleaning filtered signal Power Spectrum')
plt.show ()

#%%

#Somehow working but dont know what is been plotted
f0=1.5
N =int(500000* 7.6)   #Sampling Rate * Duration

yf = np.fft.fft(y_2_bp_kaiser)
xf = np.fft.fftfreq(N, 1 / 500000)

plt.xlim(-10, 10)
plt.plot(xf, np.abs(yf))
plt.show()


# %%
# Fast Fourier Transformation of input and output signals 
f0=2
N = int(10*(fs/f0))   

yf_input = np.fft.rfft(sig) 
y_input_mag_plot = np.abs(yf_input)/N


f= np.linspace (0, (N-1)*(fs/N),N )

f_plot = f[0:int(N/2+1)]
y_input_mag_plot = 2*y_input_mag_plot[0:int(N/2+1)]
y_input_mag_plot[0] = y_input_mag_plot[0] / 2

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
fig.suptitle('IIR butterworth Low-pass Filter')
ax1.plot(f_plot, y_input_mag_plot)
ax1.grid(alpha=0.3)

ax1.set_ylabel('Amplitute [dB]')


yf_output = np.fft.rfft(y_2_bp_kaiser)
y_output_mag_plot = np.abs(yf_output)/N
y_output_mag_plot = 2* y_output_mag_plot[0:int(N/2+1)]
y_output_mag_plot[0]= y_output_mag_plot[0]/2  
ax2.plot(f_plot, y_output_mag_plot)

plt.plot(f_plot, y_output_mag_plot, label='output signal')
plt.grid(alpha=0.3)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitute [dB]')
plt.ylim(-1, 1)
plt.xscale('log')
plt.legend()

plt.show()

#%%
#Plot the signal in frequency domain after using FIR LP (signal.remez) 
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
fig.suptitle('FIR Low-pass Filter (signal.remez)')
ax1.plot(f_plot, y_input_mag_plot, label= 'input signal')
ax1.grid(alpha=0.3)

ax1.set_ylabel('Amplitute [dB]')

yf_output = np.fft.rfft(y_bp_base_clear)
y_output_mag_plot = np.abs(yf_output)/N
y_output_mag_plot = 2* y_output_mag_plot[0:int(N/2+1)]
y_output_mag_plot[0]= y_output_mag_plot[0]/2  
ax2.plot(f_plot, y_output_mag_plot)

plt.plot(f_plot, y_output_mag_plot, label= 'output signal')
plt.grid(alpha=0.3)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitute [dB]')
plt.xscale('log')
plt.ylim(-1, 2)
plt.legend()

plt.show()



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
