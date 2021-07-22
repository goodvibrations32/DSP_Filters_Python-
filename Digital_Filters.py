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



