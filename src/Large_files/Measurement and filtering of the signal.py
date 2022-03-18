
# %%
#Use of pandas library for reading hdf5 file format


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from pylab import figure, plot, grid, show, ylim, title, legend
#%%
# Define function for Filter freq response

def plot_response(fs, w, h, title):
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    plt.ylim(-40, 5)
    #plt.xscale('log')
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title(title)

#%%
#Define a function for plotting the Power spectrums

def plot_spectrum(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.semilogy(x, np.sqrt(y))
    plt.grid(True, which='both')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


#%%
#Define function for the FFT plot
def plot_FFT (x1, y1, y2, title, xlabel, ylabel):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.suptitle(title)
    ax1.semilogx(x1, y1)
    ax1.grid(True, which = 'both')
    ax1.set_ylabel(ylabel)
    plt.semilogx(x1, y2, 'orange', label = 'output')
    plt.grid(True, which='both')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 0.025)
    plt.legend()
    plt.show()


#%%
#Define function for FFT
def fft_sig (y1, y2):
    f0 = 2000
    fs = 500000
    N = int(2000*(fs/f0))
    yf_input = np.fft.fft(y1)
    y_input_mag_plot = np.abs(yf_input)/N
    f= np.linspace (0, (N-1)*(fs/N),N )
    f_plot = f[0:int(N/2+1)]
    y_input_mag_plot = 2*y_input_mag_plot[0:int(N/2+1)]
    y_input_mag_plot[0] = y_input_mag_plot[0] / 2

    yf_output = np.fft.fft(y2)
    y_output_mag_plot = np.abs(yf_output)/N
    y_output_mag_plot = 2* y_output_mag_plot[0:int(N/2+1)]
    y_output_mag_plot[0]= y_output_mag_plot[0]/2
    return(f_plot, y_input_mag_plot, y_output_mag_plot)


#%%
#Define function to plot the raw and filtered signals combined 

def plot_signals(x1,x2,y1,y2,Title):
    title(Title)
    plot(x1, y1, label= 'Original signal')
    plot(x2, y2, label='Filtered signal')
    grid(True)
    legend(loc='lower right')
    show()


#%%
#Read and store the .h5 file with pandas
f_1 = pd.HDFStore(path='/run/media/goodvibrations/KINGSTON/_noiseReference_2021/20210831/noise_reference_raw.h5', mode='r')


data_raw = f_1['/df']

first_column =data_raw.get('raw-2021.08.31-12:26:54')

first_column.info()

second_column = data_raw.get('raw-2021.08.31-12:28:24')

third_column = data_raw.get('raw-2021.08.31-12:32:08')

fourth_column = data_raw.get('raw-2021.08.31-12:34:29')

fifth_column = data_raw.get('raw-2021.08.31-12:35:42')

sixth_column = data_raw.get('raw-2021.08.31-12:37:45')

#%%
time = np.linspace(0, 7.599998, 3800000)

sig_inverter_con_off = np.array(first_column)

sig_inverter_con_on = np.array(second_column)

sig_inverter_con_on_WS_5 = np.array(third_column)

sig_inverter_discon_off = np.array(fourth_column)

sig_inverter_discon_on = np.array(fifth_column)

sig_inverter_discon_on_WS_5 = np.array(sixth_column)

plt.plot(time, sig_inverter_con_off)
plt.show()

print (first_column.head())

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
cutoff_hz = 150
nyq_rate = fs/2 


#Use of firwin to create low-pass FIR filter
fir_co = signal.firwin(numtaps_2, cutoff_hz/nyq_rate)
w_fir_co, h_fir_co = signal.freqz(fir_co, [1])


#%%
#Plot the freq response of the filter
plot_response(fs, w_fir_co, h_fir_co, 'Blank FIR filter')


#%%
#Apply the filter to the signal
blank_lp_fir_con_off = signal.lfilter(fir_co, 1.0, sig_inverter_con_off)

blank_lp_fir_con_on = signal.lfilter(fir_co, 1.0, sig_inverter_con_on)

blank_lp_fir_con_on_WS_5 = signal.lfilter(fir_co, 1.0, sig_inverter_con_on_WS_5)

blank_lp_fir_discon_off = signal.lfilter(fir_co, 1.0, sig_inverter_discon_off)

blank_lp_fir_discon_on = signal.lfilter(fir_co, 1.0, sig_inverter_discon_on)

blank_lp_fir_discon_on_WS_5 = signal.lfilter(fir_co, 1.0, sig_inverter_discon_on_WS_5)


#%%
#=====================================================
#plot original and filtered signal different approach
#=====================================================

#The first N-1 samples are corrupted by the initial conditions
warmup = numtaps_2 - 1

#The phase delay of the filtered signal
delay= (warmup / 2) / fs

#figure(1)
#Plot the original (unfiltered) signal
#plot(time, sig)

#Title of green line
title('Filtered signal with phase shift with Inverter connected and off')

#Plot the filtered signal (filter output) with phase shift
plot(time-delay, blank_lp_fir_con_off, 'g-')

grid(True)

ylim((1.585,1.595))

show()


#figure(2)

#Title of original and 0 phase signal (red line)
title('Original signal Inverter connected and off')
time_no_shift = time[warmup:]-delay
filt_con_off  = blank_lp_fir_con_off[warmup:]

#Plot the gooood stuff (no corruption)

plot(time, sig_inverter_con_off)
grid(True)
show()

title('Filtered signal Inverter connected and off')
plot(time_no_shift, filt_con_off, 'r')
grid(True)
#ylim((1.585,1.595))
show()

#figure(3)

title('Original signal Inverter connected and on')

filt_con_on = blank_lp_fir_con_on[warmup:]
plot(time, sig_inverter_con_on)
grid(True)
show()

title('Filtered signal Inverter connected and on')
plot(time_no_shift,filt_con_on, 'y')

grid(True)
show()

#figure(4)
title('Original signal Inverter connected, on and wind speed 5 [m/s]')
filt_con_on_WS_5 = blank_lp_fir_con_on_WS_5[warmup:]
plot(time, sig_inverter_con_on_WS_5)
grid(True)
show()

title('Filtered signal Inverter connected, on and wind speed 5 [m/s]')
plot(time_no_shift, filt_con_on_WS_5, 'black')
grid(True)
show()

#figure(6)
title('Original signal with Inverter disconnected and off')
filt_discon_off = blank_lp_fir_discon_off[warmup:]
plot(time, sig_inverter_discon_off)
grid(True)
show()

title('Filtered signal with Inverter disconnected and off')
plot(time_no_shift, filt_discon_off, 'r-')
grid(True)
show() 

#figure (7)
title('Original signal with Inverter disconnected and on')
filt_discon_on = blank_lp_fir_discon_on[warmup:]
plot(time, sig_inverter_discon_on)
grid(True)
show()

title('Filtered signal with Inverter disconnected and on')
plot(time_no_shift, filt_discon_on, 'y')
grid(True)
show()

#figure(8)
title('Original signal with Inverter disconnected, on and Wind speed 5 [m/s]')
filt_discon_on_WS_5 = blank_lp_fir_discon_on_WS_5[warmup:]
plot(time, sig_inverter_discon_on_WS_5)
grid(True)
show()

title('Filtered signal with Inverter disconnected, on and Wind speed 5 [m/s]')
plot(time_no_shift, filt_discon_on_WS_5, 'black')
grid(True)
show()

#%%
#Plot of the raw and filtered signals in one graph with different colors for 
#better understanding the response of the system in time domain
# 
#  

#===============================================
#Inverter connected and off
#===============================================
#figure(1)
time_no_shift = time[warmup:]-delay
filt_con_off  = blank_lp_fir_con_off[warmup:]
plot_signals(time,time_no_shift,sig_inverter_con_off,filt_con_off,'Inverter connected and off')

#===============================================
#Inverter connected and on
#===============================================
#figure(2)
filt_con_on = blank_lp_fir_con_on[warmup:]
plot_signals(time,time_no_shift,sig_inverter_con_on,filt_con_on,'Inverter connected and on')

#===============================================
#Inverter connected, on and WS=5[m/s]
#===============================================
#figure(3)
filt_con_on_WS_5 = blank_lp_fir_con_on_WS_5[warmup:]
plot_signals(time,time_no_shift,sig_inverter_con_on_WS_5,filt_con_on_WS_5,'Inverter connected, on and wind speed 5 [m/s]')

#===============================================
#Inverter disconnected and off
#===============================================
#figure(4)
filt_discon_off = blank_lp_fir_discon_off[warmup:]
plot_signals(time,time_no_shift,sig_inverter_discon_off,filt_discon_off,'Inverter disconnected and off')

#===============================================
#Inverter disconnected and on
#===============================================
#figure (5)
filt_discon_on = blank_lp_fir_discon_on[warmup:]
plot_signals(time,time_no_shift,sig_inverter_discon_on,filt_discon_on,'Inverter disconnected and on')

#====================================================
#Inverter disconnected and on with wind speed = 5 m/s
#====================================================
#figure(6)
filt_discon_on_WS_5 = blank_lp_fir_discon_on_WS_5[warmup:]
plot_signals(time,time_no_shift,sig_inverter_discon_on_WS_5,filt_discon_on_WS_5,'Inverter disconnected, on and Wind speed 5 [m/s]')

#////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////

# %%
#===============================================
#Inverter connected and off
#===============================================

f, Pxx_spec = signal.welch(sig_inverter_con_off, fs, window ='flattop', nperseg = 1024, scaling = 'spectrum')
plot_spectrum(f, Pxx_spec,'Raw signal Power spectrum (Inverter connected and off)', 'Frequency [Hz]', 'Linear spectrum [V RMS]')

# %%
#===============================================
#Inverter connected and off
#===============================================

f, Pxx_spec = signal.welch(filt_con_off, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Filtered signal Power spectrum (Inverter connected and off)', 'Frequency [Hz]', 'Linear spectrum [V RMS]' )

#%%
#===============================================
#Inverter connected and on
#===============================================

f, Pxx_spec = signal.welch(sig_inverter_con_on, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Raw signal Power spectrum (Inverter connected and on)', 'Frequency [Hz]', 'Linear spectrum [V RMS]' )

# %%
#===============================================
#Inverter connected and on
#===============================================

f, Pxx_spec = signal.welch(filt_con_on, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Filtered signal Power spectrum (Inverter connected and on)', 'Frequency [Hz]', 'Linear spectrum [V RMS]')

#%%
#===============================================
#Inverter connected, on and WS=5[m/s]
#===============================================

f, Pxx_spec = signal.welch(sig_inverter_con_on_WS_5, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Raw signal Power spectrum (Inverter connected, on and WS=5m/s)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]' )

# %%
#===============================================
#Inverter connected, on and WS=5[m/s]
#===============================================

f, Pxx_spec = signal.welch(filt_con_on_WS_5, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Filtered signal Power Spectrum (Inverter connected, on and WS=5m/s)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]')

# %%
#===============================================
#Inverter disconnected and off
#===============================================

f, Pxx_spec = signal.welch(sig_inverter_discon_off, fs, window ='flattop', nperseg = 1024, scaling = 'spectrum')
plot_spectrum(f, Pxx_spec, 'Raw signal Power spectrum (Inverter disconnected and off)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]')

# %%
#===============================================
#Inverter disconnected and off
#===============================================

f, Pxx_spec = signal.welch(filt_discon_off, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Filtered signal Power Spectrum (Inverter disconnected and off)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]' )


#%%
#===============================================
#Inverter disconnected and on
#===============================================

f, Pxx_spec = signal.welch(sig_inverter_discon_on, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Raw signal Power spectrum (Inverter disconnected and on)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]')

#%%
#===============================================
#Inverter disconnected and on
#===============================================

f, Pxx_spec = signal.welch(filt_discon_on, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Filtered signal Power Spectrum (Inverter disconnected and on)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]')


#%%
#====================================================
#Inverter disconnected and on with wind speed = 5 m/s
#====================================================

f, Pxx_spec = signal.welch(sig_inverter_discon_on_WS_5, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Raw signal Power Spectrum (Inverter disconnected, on and WS=5m/s)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]')

#%%
#====================================================
#Inverter disconnected and on with wind speed = 5 m/s
#====================================================

f, Pxx_spec = signal.welch(filt_discon_on_WS_5, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plot_spectrum(f, Pxx_spec, 'Filtered signal Power Spectrum (Inverter disconnected, on and WS=5m/s)', 'Frequency [Hz]', 'Linear Spectrum [V RMS]')

#%%
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



# %%
#Fourier transform for plotting the signals original and filtered in the frequency domain

#>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>
#FOR_CONNECTED_INVERTER>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>

#===============================================
#Inverter connected, off 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_con_off, blank_lp_fir_con_off)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Blank filter output with Inverter connected, off', 'Frequency [Hz]', 'Amplitute [dB]' )

#%%
#===============================================
#Inverter connected, off 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_con_off, filt_con_off)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Not shifted filter output with Inverter connected, off', 'Frequency [Hz]', 'Amplitute [dB]' )


# %%
#===============================================
#Inverter connected, on 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_con_on, blank_lp_fir_con_on)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Blank filter output with Inverter connected, on', 'Frequency [Hz]', 'Amplitute [dB]' )


#%%
#===============================================
#Inverter connected, on 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_con_on, filt_con_on)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Not shifted filter output with Inverter connected, on', 'Frequency [Hz]', 'Amplitute [dB]' )


#%%
#===============================================
#Inverter connected, on and WS=5[m/s]
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_con_on_WS_5, blank_lp_fir_con_on_WS_5)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Blank filter output with Inverter connected, on and WS=5[m/s]', 'Frequency [Hz]', 'Amplitute [dB]' )


#%%
#===============================================
#Inverter connected, on and WS=5[m/s]
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_con_on_WS_5, filt_con_on_WS_5) 
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Not shifted filter output with Inverter connected, on and WS=5[m/s]', 'Frequency [Hz]', 'Amplitute [dB]' )


# %%

#>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>
#FOR_DISCONNECTED_INVERTER>> 
#>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>

#===============================================
#Inverter disconnected, off 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_discon_off, blank_lp_fir_discon_off)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Blank filter output with Inverter disconnected, off', 'Frequency [Hz]', 'Amplitute [dB]' )


#%%
#===============================================
#Inverter disconnected, off 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_discon_off, filt_discon_off)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Not shifted filter output with Inverter disconnected, off', 'Frequency [Hz]', 'Amplitute [dB]' )


# %%
#===============================================
#Inverter disconnected, on 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_discon_on, blank_lp_fir_discon_on)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Blank filter output with Inverter disconnected, on ', 'Frequency [Hz]', 'Amplitute [dB]' )


#%%
#===============================================
#Inverter disconnected, on 
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_discon_on, filt_discon_on)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Not shifted filter output with Inverter disconnected, on ', 'Frequency [Hz]', 'Amplitute [dB]' )


#%%
#===============================================
#Inverter disconnected, on and WS=5[m/s]
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_discon_on_WS_5, blank_lp_fir_discon_on_WS_5)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Blank filter output with Inverter disconnected, on and WS=5[m/s]', 'Frequency [Hz]', 'Amplitute [dB]' )


#%%
#===============================================
#Inverter disconnected, on and WS=5[m/s]
#===============================================

f_plot, y_input_mag_plot, y_output_mag_plot= fft_sig(sig_inverter_discon_on_WS_5, filt_discon_on_WS_5)
plot_FFT(f_plot, y_input_mag_plot, y_output_mag_plot, 'Not shifted filter output with Inverter disconnected, on and WS=5[m/s]', 'Frequency [Hz]', 'Amplitute [dB]' )


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# %%
