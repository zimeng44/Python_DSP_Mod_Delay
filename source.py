import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio
import librosa

def speeder(X, rate, n_h): # phase vocoder for audio time expansion/compression

    frames = np.arange(0, X.shape[-1], rate) # len(frames)=the number of frames for the output, value of each element is the relative position of a frame in X_mod in X

    s = list(X.shape) #get the dimensions of the input signal for the output signal
    s[-1] = len(frames) # the output signal has the same number of rows as the input signal but different number of columns 
    X_mod = np.zeros(s,dtype=complex) # make the output array with the new shape

    phase_acc = np.angle(X[..., 0]) # start point: first frame
    phi_advance = np.linspace(0, np.pi * n_h, X.shape[-2]) # phase variation of frequency bins between any two frames

    padding = [(0, 0) for _ in X.shape] # padding = [(0,0),(0,0)] -> (0,0) for each row, (0,0) for each column
    padding[-1] = (0, 2) #for each row add 0 zero in the begning and 2 zeros at the end
    X = np.pad(X, padding, mode="constant") #the actually process of padding 2 zeros at the end of each row

    for i in range(0,len(frames)-2): # calculate value for each frame in output
        columns = X[..., int(frames[i]) : int(frames[i]+2)] # take out two adjacent frames from X at scaled indices(stored in the 'frames' array)
        scale = np.mod(frames[i], 1.0) # where the frame index of output falls between the two frames in the input X
        mag = (1.0 - scale) * np.abs(columns[..., 0]) + scale * np.abs(columns[..., 1]) # calculate magnitude according the position from the line above
        X_mod[..., i] = mag*(np.cos(phase_acc) + 1j * np.sin(phase_acc)) # put magnitude and phase together into X_mod
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance # phase change between the two frames minus the phase of the analysis frequency
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi)) # Wrap to -pi:pi range
        phase_acc += phi_advance + dphase # for next frame in the output
    return X_mod

def delay_cgt(loop_index, d_samples, s_rate): #since the echo is accelerated, the delay time for each echo is also accelerated
    result = 0
    for i in range(loop_index): # the delay time is aggregated as each echo happens
        result += int(d_samples / (s_rate**i))
    return result

def echo_mod(file, delay_time, amp_rate, speedup_rate, ratio): # input signal, delay time, gain, acceleration rate, wet/dry ratio
    x, index = librosa.effects.trim(file[0]) # import file and trim silence before and after
    sr = file[1] # sample rate
    delay_samples = int(delay_time * sr) # convert delay time to delay samples
    
    out_matrix = np.zeros((1, len(x))) #create a output matrix: each row is a delayed version of the original signal
    out_matrix[0,:] = x/np.max(np.abs(x)) #only need one row for now

    if amp_rate <= 0 or speedup_rate < 1: # since this is a sped-up echo function, acceleration rate should be larger than one, gain should be larger than 1 too to create a crescendo/riser feel. 
       return print('Input Error!')
    else:
        i = 0
        while (delay_samples/(speedup_rate**i)) > 0.01*sr: #when the accelerated delay time is not too small, new wcho can be created. 
            out_matrix = np.append(out_matrix, np.zeros((1, out_matrix.shape[1])), axis=0) # add a new row
            st_last = delay_cgt(i,delay_samples,speedup_rate) #locate where the accelerated dry sigal starts in the last round of delay
            en_last = len(out_matrix[i,...]) #locate where the accelerated dry sigal ends in the last round of delay
            st_now = delay_cgt(i+1,delay_samples,speedup_rate) #locate where the accelerated dry sigal starts in the current round of delay

            X = librosa.stft(out_matrix[i , st_last : ], n_fft=2048, hop_length=512) # time stretch the signal in last round of delay
            X_mod  = speeder(X, speedup_rate, 512)
            x_mod  = librosa.istft(X_mod, hop_length=512)
            en_now = st_now + len(x_mod) # #locate where the accelerated dry sigal ends in the current round of delay

            len_add = en_now - en_last # determine how many new columns needed for the new row
            if len_add >0: # in case, at some point, the new row might be accelerated so much that it's even shorther than the signal in the last round of delay
                out_matrix = np.append(out_matrix, np.zeros((out_matrix.shape[0], len_add)), axis=1) # add new columns for the current round of delay
            
            out_matrix[i+1, st_now : st_now + len(x_mod)] = amp_rate * x_mod # put the time stretched signal into the matrix
            i += 1
        
        wet = out_matrix[1,...] # sum up all rows except the first one
        for n in range(2,out_matrix.shape[0]):
            wet += out_matrix[n,...]

        output = (1-ratio) * out_matrix[0,...] + ratio * wet # sum up the dry and wet signal with ratio
        output_norm = output/np.max(np.abs(output)) #normalize the output
        return output_norm