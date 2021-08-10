# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import math
import matplotlib.pyplot as plt 
from scipy import signal

# 4TH ORDER BUTTERWORTH FILTER WITH A GAIN DROP OF 1/sqrt(2) AT 0.4 CYCLES/SAMPLE
bb, ab  = signal.butter (4, 0.8, 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab)
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))

plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which = 'both', axis='both')
plt.savefig('Butterworth Filter Freq Response.png')


# %%
# 4TH ORDER BESSEL FILTER WITH A GAIN DROP OF 1/sqrt(2) AT 0.4 CYCLES/SAMPLE

bb, ab = signal.bessel (4, 0.8, 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab)
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))

plt.title('Bessel filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')
plt.savefig('Bessel Filter Freq Response.png')


# %%
#4TH ORDER CHEBYSHEV FILTER TYPE 1 (ONLY IN PASSBAND RIPPLES) WITH MAX RIPPLES=2 AND THE GAIN DROP AT 1.5 CYCLES/SAMPLE

bb, ab = signal.cheby1 (4, 2, 0.3, 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab)
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))

plt.title('Chebyshev filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')
plt.savefig('Chebyshev Filter Freq Response.png')


# %%
# 4TH ORDER ELLIPTIC FILTER WITH MAX RIPPLES =2dB IN PASSBAND, MIN ATTENUATION =8dB IN STOP BAND AT 0.25 CYCLES/SAMPLE

bb, ab = signal.ellip (4, 2, 8, 0.5, 'low', analog=False, output='ba')
print ('Coefficients of b = ', bb)
print ('Coefficients of a = ', ab)
wb, hb = signal.freqz(bb, ab)
wb = wb/(2*math.pi)
plt.plot(wb, abs(np.array(hb)))

plt.title('Elliptic filter frequency response')
plt.xlabel('Frequency [cycles/sample]')
plt.ylabel('Amplitute [dB]')
plt.margins(0, 0.1)
plt.grid(which= 'both', axis= 'both')
plt.savefig('Elliptic Filter Freq Response.png')


# %%
#Apllying a filter 
def generate_random_signal():
    t= linspace(0, 1, 1000, False) # 1 sec
    sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.random.rand(t.shape[0])
    return (t, sig)
t, sig = generate_random_signal()

sos = signal.butter(10, 30, 'lp', fs = 1000, output = 'sos')
filtered = signal.sosfilt(sos, sig)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex= True)
fig.suptitle('Filtering of signal with f1 = 10 [Hz], f2 = 20 [Hz] and noise')
ax1.plot(t, sig)
ax1.set_title('10 and 20 Hz sinusoids + noise')
ax1.axis([0, 1, -2, 2])
ax2.plot(t, filtered)
ax2.set_title('After Filtering the noise')
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()

# %%
# Power Spectrum / signal.welch
fs = 1000
f, Pxx_spec = signal.welch(sig, fs, 'flattop', 1024, scaling = 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum (scipy.signal.welch)')
plt.show()

# %%
#filtered signal Power Spectrum
fs = 1000
f, Pxx_spec = signal.welch(filtered, 'flattop', 1000, scaling= 'spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel ('Frequency [Hz]')
plt.ylabel ('Linear Spectrum [V RMS]')
plt.title ('Power Spectrum (scipy.signal.welch)')
plt.show ()
