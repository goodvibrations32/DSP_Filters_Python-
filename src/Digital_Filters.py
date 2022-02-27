# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import math
import matplotlib.pyplot as plt 
from scipy import signal

# 4TH ORDER BUTTERWORTH FILTER WITH A GAIN DROP OF 1/sqrt(2) AT 0.4 CYCLES/SAMPLE
bb, ab  = signal.butter (N = 4,Wn = 0.8, btype= 'low', analog=False, output='ba')
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

bb, ab = signal.bessel (N = 4, Wn = 0.8, btype = 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab, worN = 512, whole = False, include_nyquist = True)
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))

plt.title('Bessel filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')

# %%
#4TH ORDER CHEBYSHEV FILTER TYPE 1 (ONLY IN PASSBAND RIPPLES) WITH MAX RIPPLES=2 AND THE GAIN DROP AT 1.5 CYCLES/SAMPLE

bb, ab = signal.cheby1 (N = 4, rp = 2, Wn = 0.3, btype = 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab, worN = 512, whole = False, include_nyquist = True)
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))

plt.title('Chebyshev filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')

# %%
# 4TH ORDER ELLIPTIC FILTER WITH MAX RIPPLES =2dB IN PASSBAND, MIN ATTENUATION =8dB IN STOP BAND AT 0.25 CYCLES/SAMPLE

bb, ab = signal.ellip (N = 4, rp = 2, rs = 8, Wn = 0.5, btype = 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab, worN = 512, whole = False, include_nyquist = True)
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))

plt.title('Elliptic filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')


# %%
#Apllying a filter on a sinusoidal signal with random noise and retrieving the pure sinusoidal tone of the sum of sin waves 10 Hz and 20 Hz 
def generate_random_signal():
    t= np.linspace(0, 1, 1000, False) # 1 sec
    sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.random.rand(t.shape[0])
    return (t, sig)
t, sig = generate_random_signal()

sos = signal.butter(N = 10, Wn = 30, btype = 'lp', fs = 1000, output = 'sos')
filtered = signal.sosfilt(sos, sig)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex= True)
fig.suptitle('Filtering of signal with f1 = 10 [Hz], f2 = 20 [Hz] and noise')
ax1.plot(t, sig)
ax1.set_title('10 and 20 Hz sinusoids + noise')
ax1.axis([0, 1, -3, 3])
ax2.plot(t, filtered)
ax2.set_title('After Filtering the noise')
ax2.axis([0, 1, -3, 3])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()

# %%
# Power Spectrum / signal.welch
fs = 1000
f, Pxx_spec = signal.welch(sig, fs, window ='flattop', nperseg = 1000, scaling = 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum (scipy.signal.welch)')
plt.show()

# %%
#filtered signal Power Spectrum
fs = 1000
f, Pxx_spec = signal.welch(filtered, fs, window ='flattop', nperseg = 1000, scaling= 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel ('Frequency [Hz]')
plt.ylabel ('Linear Spectrum [V RMS]')
plt.title ('Power Spectrum (scipy.signal.welch)')
plt.show ()

#%%

