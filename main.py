
import socket   # if error here, "pip install socket" in anaconda prompt
import numpy as np
import sounddevice as sd # if error here, "pip install sounddevice" in anaconda prompt
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, freqz
from scipy.fft import fft, ifft


SAMPLING_FREQUENCY = 40000


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
    delays_mat = np.zeros((5))
    for iChannel in range(5):
        ch1 = np.fft.fft(channels_ds[:, iChannel])
        ch2 = np.conj(np.fft.fft(channels_ds[:, 0]))
        autocorrelatedSignal = np.fft.fftshift(np.fft.ifft(ch1 * ch2))
        # autocorrelatedSignal = autocorrelatedSignal[int(autocorrelatedSignal.shape[0]/2) - DELAY_INTERVAR : int(autocorrelatedSignal.shape[0]/2) + DELAY_INTERVAR]
        delays_mat[iChannel] = int(autocorrelatedSignal.shape[0]/2) - np.argmax(autocorrelatedSignal)
    
    delays_mat_hor = np.tile(delays_mat, (5, 1))
    delays_mat_ver = delays_mat_hor.T
    delays = delays_mat_ver  - delays_mat_hor
    return np.reshape(delays, (1, -1))


def find_AngleOfArrival(delays, steering):
    pattern = np.zeros((180))
    for theta in range(180):
        pattern[theta] = np.sum(abs(delays[0, :]-steering[:, theta]))

    AngleOfArrival = np.argmin(pattern)+1
    return AngleOfArrival, pattern

VAD_THRESHOLD = 70  # minimum energy needed for diferentiating between signal and noise
FRAME_DURATION = 3e-3
FRAME_OVERLAP_DURATION = 1e-3
FRAME_LENGTH = int(FRAME_DURATION * SAMPLING_FREQUENCY)
FRAME_OVERLAP_LENGTH = int(FRAME_OVERLAP_DURATION * SAMPLING_FREQUENCY)
def VAD(channel_ds):
    ind = 0
    frames = []
    while ind + FRAME_LENGTH < channel_ds.shape[0]:
        
        channels_test = channel_ds[ind:ind + FRAME_LENGTH, :]
        enengy_test = np.sum(channels_test ** 2, axis=0)
        enengy_test = enengy_test[0:5]
        # print("energy of 1st channel:   ", enengy_test[0])
        energy_check = enengy_test > VAD_THRESHOLD
        frame_check = np.all(energy_check)
        frame = [channels_test, frame_check]
        frames.append(frame)
        ind = ind + FRAME_LENGTH - FRAME_OVERLAP_LENGTH
        
    return frames

VAD2_THRESHOLD = 1
def VAD2(channel_ds):
    channel_test = channel_ds[:, 0]
    ind = np.argwhere(channel_test > VAD2_THRESHOLD)
    ind_min = np.min(ind)
    frame = channel_ds[ind_min-30 : ind_min + 100, :]
    
    return [[frame, True]]
# from struct import 

# creating a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("192.168.1.15", 5000)
sock.bind(address)

ORDER = 20
# fs = 40000       # sample rate, Hz
STOP_FREQUENCY = 5000  # desired cutoff frequency of the filter, Hz
SOUND_VELOCITY =  343; # in m/s
MIC_POSITION  = np.array([list(range(-2,3)) , np.zeros(5)]) * 0.061 
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
        for kTheta in range(len(theta)):
            steering_mat[iChannel,jChannel, int(kTheta)] = -np.round((SAMPLING_FREQUENCY/SOUND_VELOCITY) * np.dot(np.tile(aa, (vMat.shape[0], 1))[int(kTheta), :], vMat[int(kTheta), :]))
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
                scipy.io.savemat('channel' + str(iChannel) + '.mat', {'mydata': channel_ds[:, iChannel]}) # saving file for reading in MATLAB
            channel_ds = channel_ds[51:, :]
            
            # frames = VAD(channel_ds)
            frames = VAD2(channel_ds)
            AngleOfArrival_all = []
            for frame in frames:
                if frame[1]:
                    channel_test = frame[0]
                    delays = find_delays(channel_test)
                    AngleOfArrival, pattern = find_AngleOfArrival(delays, steering)
                    # print('AngleOfArrival:    ', AngleOfArrival)
                    AngleOfArrival_all.append(AngleOfArrival)  
                    
            print(AngleOfArrival_all)        
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