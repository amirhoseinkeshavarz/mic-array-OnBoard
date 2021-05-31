# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:40:05 2021

@author: ASUS
"""

import socket   # if error here, "pip install socket" in anaconda prompt
import numpy as np
import sounddevice as sd # if error here, "pip install sounddevice" in anaconda prompt
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# from struct import 

# creating a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("192.168.1.15", 5000)
sock.bind(address)

SAMPLING_FREQUENCY = 40000

channel_acc = np.zeros((1,8))
counter = 0
while True:
    try:
        print(counter)
        counter += 1
        if counter > 2000: # this specifies how many packet should put together and then played
            break
        
        rawData, server = sock.recvfrom(10000) # reading form socket. 10000 is buffer length 
        rawData = np.array([rawData[18:818]]) # discarding headers.
        dt = np.dtype('uint16')
        dt = dt.newbyteorder('>')
        data1 = np.frombuffer(rawData,dtype= dt) # converting to uint16
        channels = data1.reshape(-1, 8) # reshaping, so every channel gets in a colomn
        channel = channels # reading channel 2
        # channel = channels[:, 4] # reading channel 2

        # channel = channel - channel.mean() # eliminating DC
        channel = channel / 2**10 # getting a float number between 0 and 1
        channel_acc = np.concatenate((channel_acc, channel), axis=0) # putting together packets

    except Exception as err:
        print(err)
        sock.close()
        break

ds = 1 # DOWN SAMPLE rate
channel_ds = channel_acc[::ds]
channel_ds = channel_ds - channel_ds.mean()

order = 20
fs = 40000       # sample rate, Hz
cutoff = 5000  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)
plt.figure()
for iChannel in range(6):
    channel_ds_ch = channel_ds[:, iChannel+0]
    channel_ds_ch = butter_lowpass_filter(channel_ds_ch, cutoff, fs, order)
    channel_ds_ch = channel_ds_ch[50:]
    freq = np.linspace(-fs/2,fs/2,channel_ds_ch.shape[0])
    channel_spectrum = np.fft.fft(channel_ds_ch)
    channel_spectrum = np.fft.fftshift(channel_spectrum)
    cahnnel_spectrum_log = np.log10(channel_spectrum)

    # plt.figure()
    # plt.plot(freq,cahnnel_spectrum_log)
    
    plt.plot(channel_ds_ch)
    plt.show()
plt.legend(['1','2','3','4','5','6'])
sd.play(channel_ds_ch, SAMPLING_FREQUENCY/ds) # playing
# status = sd.wait()

# scipy.io.savemat('test.mat', {'mydata': channel_acc}) # saving file for reading in MATLAB
# scipy.io.wavfile.write('test.wav', fs, channel_ds)
    
        