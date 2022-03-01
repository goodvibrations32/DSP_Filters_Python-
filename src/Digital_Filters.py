# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from cmath import log10, pi
from matplotlib import projections
import numpy as np 
from numpy import logspace
import math
import matplotlib.pyplot as plt
from scipy import signal

# 4TH ORDER BUTTERWORTH FILTER WITH A GAIN DROP OF 1/sqrt(2) AT 0.4 CYCLES/SAMPLE
bb, ab  = signal.butter (N = 10,Wn = 0.8, btype= 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab, worN = 512, whole = False, include_nyquist = True) # adding "include_nyquist = True" plots the last frequency that is otherwise ignored if "worN = int" && "whole = False" 
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))  

plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which = 'both', axis='both')



# %%
# 4TH ORDER BESSEL FILTER WITH A GAIN DROP OF 1/sqrt(2) AT 0.4 CYCLES/SAMPLE

bb_1, ab_1 = signal.bessel (N = 4, Wn = 0.8, btype = 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb_1)
print ('Coefficients of a = ', ab_1)
wb_1, hb_1 = signal.freqz(bb_1, ab_1, worN = 512, whole = False, include_nyquist = True)
wb_1 = wb_1/(2*math.pi)
plt.plot(wb_1, abs(np.array(hb_1)))

plt.title('Bessel filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')

# %%
#4TH ORDER CHEBYSHEV FILTER TYPE 1 (ONLY IN PASSBAND RIPPLES) WITH MAX RIPPLES=2 AND THE GAIN DROP AT 1.5 CYCLES/SAMPLE

bb_2, ab_2 = signal.cheby1 (N = 4, rp = 2, Wn = 0.3, btype = 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb_2)
print ('Coefficients of a = ', ab_2)
wb_2, hb_2 = signal.freqz(bb_2, ab_2, worN = 512, whole = False, include_nyquist = True)
wb_2 = wb_2/(2*math.pi)
plt.plot(wb_2, abs(np.array(hb_2)))
plt.title('Chebyshev filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')

# %%
# 4TH ORDER ELLIPTIC FILTER WITH MAX RIPPLES =2dB IN PASSBAND, MIN ATTENUATION =8dB IN STOP BAND AT 0.25 CYCLES/SAMPLE

bb_3, ab_3 = signal.ellip (N = 4, rp = 2, rs = 8, Wn = 0.5, btype = 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb_3)
print ('Coefficients of a = ', ab_3)
wb_3, hb_3 = signal.freqz(bb_3, ab_3, worN = 512, whole = False, include_nyquist = True)
wb_3 = wb_3/(2*math.pi)
plt.plot(wb_3, abs(np.array(hb_3)))

plt.title('Elliptic filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')


# %%
#Apllying a filter on a sinusoidal signal with random noise and retrieving the pure sinusoidal tone of the sum of sin waves 10 Hz and 20 Hz 
def generate_random_signal():
    t= np.linspace(0, 1, 1024, False) # 1 sec
    f0= 10
    sig = 2* np.sin(2*np.pi*f0*t) + 3* np.sin(2*np.pi*2*f0*t) #+ np.sin(2*np.pi*1000*t)           #np.random.rand(t.shape[0]
    return (t, sig)
t, sig = generate_random_signal()
sos = signal.butter(N = 10, Wn = 15, btype = 'lp', fs = 1024, output = 'sos')
filtered = signal.sosfilt(sos, sig)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex= True)
fig.suptitle('Filtering of signal with f1 = 10 [Hz], f2 = 20 [Hz] ')
ax1.plot(t, sig)
ax1.set_title('10 and 20 Hz sinusoids ')
ax1.axis([0, 1, -5, 5])
ax2.plot(t, filtered)
ax2.set_title('After Filtering the 20 Hz')
ax2.axis([0, 1, -5, 5])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()

# %%
# Power Spectrum / signal.welch
fs = 1024
f, Pxx_spec = signal.welch(sig, fs, window ='flattop', nperseg = 1024, scaling = 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlim(0,50)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum (scipy.signal.welch)')
plt.show()

# %%
#filtered signal Power Spectrum
fs = 1024
f, Pxx_spec = signal.welch(filtered, fs, window ='flattop', nperseg = 1024, scaling= 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlim(0,50)
plt.xlabel ('Frequency [Hz]')
plt.ylabel ('Linear Spectrum [V RMS]')
plt.title ('Power Spectrum (scipy.signal.welch)')
plt.show ()

#%%
"""
# simulation of the system response to the butterworth filter used to eliminate noise
from scipy.signal import lsim
b, a = signal.butter(N=10, Wn=2*np.pi*15, btype='lowpass', analog=True)
tout, yout, xout = lsim((b, a), U=sig, T=t)
plt.plot (t, sig, 'r', alpha=0.5, linewidth=1, label='input')
plt.plot (tout, yout, 'k', linewidth=1.5, label='output')
plt.legend (loc='best', shadow=True, framealpha=1)
plt.grid (alpha=0.3)
plt.xlabel ('time')
plt.show ()
"""
# %%
# Fast Fourier Transformation of input and output signals 
from scipy.fft import rfft, rfftfreq
f0=10
N = int(10*(fs/f0))   

yf_input = np.fft.rfft(sig) 
y_input_mag_plot = np.abs(yf_input)/N


f= np.linspace (0, (N-1)*(fs/N),N )

f_plot = f[0:int(N/2+1)]
y_input_mag_plot = 2*y_input_mag_plot[0:int(N/2+1)]
y_input_mag_plot[0] = y_input_mag_plot[0] / 2

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
fig.suptitle('Frequency domain of the original signal and filtered(below)')
ax1.plot(f_plot, y_input_mag_plot)
ax1.grid(alpha=0.3)
#ax1.label('Frequency [Hz]')
ax1.set_ylabel('Amplitute [dB]')

#plt.plot (f_plot, x_mag_plot)
#plt.show()
yf_output = np.fft.rfft(filtered)
y_output_mag_plot = np.abs(yf_output)/N
y_output_mag_plot = 2* y_output_mag_plot[0:int(N/2+1)]
y_output_mag_plot[0]= y_output_mag_plot[0]/2  
ax2.plot(f_plot, y_output_mag_plot)

plt.grid(alpha=0.3)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitute [dB]')
plt.xlim(0,50)
plt.ylim(0,2)
plt.show()

# %%
#Bode plot for input and output
# needs more work for proper implementation
H= yf_output/ yf_input
Y = np.imag (H)
X = np.real (H)
#Mag_of_H = 20*cmath.log10(Y)
f = logspace(-1,0.001) # frequencies from 0 to 10**5
sys = signal.lti([H],f)
w, mag, phase = signal.bode(sys, n= f)
#print (w, mag, phase)
plt.plot(f, w, label= 'Frequency rad/sec')
plt.plot(f, mag, label= 'Magnitude [dB]' )
plt.plot(f, phase, label='Phase array [deg]' )
plt.grid(which='both', axis= 'both') 
plt.legend()       
#plt.xlabel('Real numbers R')
#plt.ylabel('Imaginary numbers I')
#plt.scatter(X,Y)


#plt.show()

#fig= plt.figure()
#ax = fig.add_subplot(projection = 'polar')
#c= ax.scatter(X,Y)
#plt.title('Polar representation of ')





# %%
