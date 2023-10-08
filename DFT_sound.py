# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:28:27 2019

@author: Ajisaroji
"""
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def naive_DFT(x):
  N = np.size(x)
  X = np.zeros((N,),dtype=np.complex128)
  for m in range(0,N):    
     for n in range(0,N): 
      X[m] += x[n]*np.exp(-np.pi*2j*m*n/N)
  return X

x, fs = sf.read('../data/speech.wav')
t = np.arange(len(x)) / fs
dt=1/fs
N=x.size  # Number of sample points
yf = naive_DFT(x)
xf = np.linspace(0.0, 1.0/(2.0*dt), N//2)

# plotting graphics  
plt.subplot(2, 1, 1); plt.plot(t, x, '-')
plt.title('Data Mentah');plt.ylabel('Amp')
plt.subplot(2, 1, 2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), 'r-')
plt.title('Power Spectral');plt.xlabel('frek (Hz)');plt.ylabel('Spectral')

plt.grid();
plt.show()


