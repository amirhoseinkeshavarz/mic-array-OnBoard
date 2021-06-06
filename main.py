# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:38:55 2021

@author: ASUS
"""

import socket   # if error here, "pip install socket" in anaconda prompt
import numpy as np
import sounddevice as sd # if error here, "pip install sounddevice" in anaconda prompt
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, freqz
from scipy.fft import fft, ifft


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

DELAY_INTERVAR = 100
def find_delays(channels_ds):
    delays_mat = np.zeros(5, 5)
    for iChannel in range(5):
        for jChannel in range(5):
            ch1 = np.fft(channels_ds[:, iChannel])
            ch2 = np.fft(channel_ds[:, jChannel])
            autocorrelatedSignal = np.ifft(ch1 * ch2)
            autocorrelatedSignal = autocorrelatedSignal[int(autocorrelatedSignal.shape[0]/2) - DELAY_INTERVAR : int(autocorrelatedSignal.shape[0]/2) + DELAY_INTERVAR]
            delays_mat[iChannel, jChannel] = int(autocorrelatedSignal.shape[0]/2) - np.argmax(autocorrelatedSignal)
    delays = np.reshape(delays_mat, (1, -1))
    return delays


def find_AngleOfArrival(delays, steering):
    pattern = np.zeros((180))
    for theta in range(180):
        pattern[theta] = np.dot(delays, steering[theta, :])

    AngleOfArrival = np.argmax(pattern)
    return AngleOfArrival, pattern

# from struct import 

# creating a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("192.168.1.15", 5000)
sock.bind(address)

SAMPLING_FREQUENCY = 40000
ORDER = 20
# fs = 40000       # sample rate, Hz
STOP_FREQUENCY = 5000  # desired cutoff frequency of the filter, Hz
MIC_POSITION  = np.array([list(range(-2,3)) , np.zeros(5)])
theta = np.linspace(0,180,180) * np.pi/180
vMat = np.array([np.cos(theta), np.sin(theta)])
vMat = vMat.T
# Get the filter coefficients so we can check its frequency response.
# b, a = butter_lowpass(STOP_FREQUENCY, SAMPLING_FREQUENCY, ORDER)
channel_acc = np.zeros((1,8))
counter = 0
processFlag = False

# creating steering vector
steering_mat = np.zeros((5, 5, 180))
for iChannel in range(5):
    for jChannel in range(5):
        aa = MIC_POSITION[:,iChannel] - MIC_POSITION[:,jChannel]
        for kTheta in theta:
            steering_mat[iChannel,jChannel, int(kTheta)] = (SAMPLING_FREQUENCY/3e8) * np.dot(np.tile(aa, (vMat.shape[0], 1))[int(kTheta), :], vMat[int(kTheta), :])
steering = np.reshape(steering_mat, (-1, steering_mat.shape[2]))

# main loop
while True:
    try:
        print(counter)
        counter += 1

        rawData, server = sock.recvfrom(40900) # reading form socket. 10000 is buffer length 
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
 

        if counter > 1000: # this specifies how many packet should put together and then played
            processFlag = True
            counter = 0
            
        if processFlag:
            ds = 1 # DOWN SAMPLE rate
            channel_ds = channel_acc[::ds]
            channel_ds = channel_ds - channel_ds.mean()


        # plt.figure()
            for iChannel in range(5):
                channel_ds[:, iChannel]= butter_lowpass_filter(channel_ds[:, iChannel], STOP_FREQUENCY, SAMPLING_FREQUENCY, ORDER)
                # channel_ds[:, iChannel] = channel_ds[50:, iChannel]
                # freq = np.linspace(-SAMPLING_FREQUENCY/2,SAMPLING_FREQUENCY/2,channel_ds_ch.shape[0])
                # channel_spectrum = np.fft.fft(channel_ds_ch)
                # channel_spectrum = np.fft.fftshift(channel_spectrum)
                # cahnnel_spectrum_log = np.log10(channel_spectrum)
                # scipy.io.savemat('channel' + str(iChannel) + '.mat', {'mydata': channel_ds_ch}) # saving file for reading in MATLAB
            channel_ds = channel_ds[50:, :]
            
            delays = find_delays(channel_ds)
            AngleOfArrival, pattern = find_AngleOfArrival(delays, steering)
            print('AngleOfArrival:    ', AngleOfArrival)
            plt.figure()
            theta = np.linspace(1, 180, 180)
            plt.plot(theta, pattern)
            plt.show()
            
            processFlag = False
            break
            
    except Exception as err:
        print(err)
        sock.close()
        break

# ds = 1 # DOWN SAMPLE rate
# channel_ds = channel_acc[::ds]
# channel_ds = channel_ds - channel_ds.mean()


# # plt.figure()
# for iChannel in range(5):
#     channel_ds_ch = channel_ds[:, iChannel+0]
#     channel_ds_ch = butter_lowpass_filter(channel_ds_ch, STOP_FREQUENCY, SAMPLING_FREQUENCY, ORDER)
#     channel_ds_ch = channel_ds_ch[50:]
#     freq = np.linspace(-SAMPLING_FREQUENCY/2,SAMPLING_FREQUENCY/2,channel_ds_ch.shape[0])
#     channel_spectrum = np.fft.fft(channel_ds_ch)
#     channel_spectrum = np.fft.fftshift(channel_spectrum)
#     cahnnel_spectrum_log = np.log10(channel_spectrum)
#     scipy.io.savemat('channel' + str(iChannel) + '.mat', {'mydata': channel_ds_ch}) # saving file for reading in MATLAB
    
    # plt.figure()
    # plt.plot(freq,cahnnel_spectrum_log)
    
    # plt.plot(channel_ds_ch)
    # plt.show()
# plt.legend(['1','2','3','4','5','6'])
# sd.play(channel_ds, SAMPLING_FREQUENCY/ds) # playing
# status = sd.wait()
# sock.close()

# for i in range(5):
#     scipy.io.savemat('channel' + str(i) + '.mat', {'mydata': channel_ds[:, i]}) # saving file for reading in MATLAB
# scipy.io.wavfile.write('test.wav', fs, channel_ds)
    
# correlated = np.correlate(channel_ds[:, 0], channel_ds[:, 1], "same")   
 
# plt.figure()
# plt.plot(correlated)
# plt.show()

# relative_ind_all = []
# for i in range(4):
#     correlated = np.fft.fftshift(ifft(fft(channel_ds[:-1, 0]) * np.conj(fft(channel_ds[:-1, i + 1]))))
    
#     max_ind = np.argmax(correlated)
#     relative_ind = max_ind - correlated.shape[0]/2
#     relative_ind_all.append(relative_ind)
#     # relative_ind = correlated.shape - argmax()
# print(relative_ind_all)

#     # plt.figure()
#     # plt.plot(abs(correlated))
#     # plt.show()

# for i in range(4):
#     print(i+1)
    

# a = fft(channel_ds[:-1, 0])
# plt.figure()
# plt.plot(np.fft.fftshift(20*np.log10(abs(a))))
# plt.show()