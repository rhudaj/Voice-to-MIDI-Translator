import numpy as np
import copy
import math
import matplotlib.pyplot as pyplot
import librosa
import AudioUtil.DataStructures.plot as plt
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.DataStructures.plot import CustomFig
from AudioUtil.MusicAnalyze.util import Envelope
from AudioUtil.MusicAnalyze.YIN import yin

def CalculateSilenceThreshold(y: np.ndarray):
    '''
        K:
            - 10% default
            - The noise/signal ratio expected in one of the poorest recording environments you'd like to support.

        Notes:
        -   Any loud noises, e.g. a hand clap, that aren't speech but have a higher amplitude,
            can cause the noise floor to raise above speech causing speech to be miss-classified as noise.
    '''
    K = 0.1
    return y.max() * 0.1


FIG = CustomFig()
ax = FIG.plot_bottom([0], [0], x_min=0, x_max=30, y_min=-1, y_max=1)

def DetectSilence(t: np.ndarray, signal: np.ndarray, threshold = 0.001):
    env = Envelope(signal, w=4096)
    ax.plot(t, env)
    #-----
    silent = False
    intervals: list[tuple[int, int]] = []
    I=[0,0]
    for i in range(env.size):
        if env[i] < threshold:
            if not silent:
                # print(f'silence start: {i}')
                # input('enter:')
                I[0] = i
                silent = True
        else:
            if silent:
                # print(f'silence end: {i-1}')
                # input('enter:')
                I[1] = i-1
                intervals.append(I)
                silent = False

    for I in intervals:
        t0 = t[I[0]]
        t1 = t[I[1]]
        # print(f'[{t0},{t1}]')
        # input('press enter')
        # ax.axvline(x=t0, color='red')
        # ax.axvline(x=t1, color='orange')
        # # signal[I[0]:I[1]] = 0

    return signal


def test():
    name = 'voiceScale_Filtered'
    AS: AudioSignal = AudioSignal_FromFile(f'../SampleInput/{name}.wav')
    y: np.ndarray = AS.signal
    t = AS.time
    fs = AS.sample_freq
    # ax.plot(t,y)
    y = DetectSilence(t, y)
    FIG.show()

test()