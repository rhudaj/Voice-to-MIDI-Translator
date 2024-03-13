import numpy as np
import librosa
import AudioUtil.DataStructures.plot as plt
import matplotlib.pyplot as pyplot
from matplotlib.axes import Axes
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.DataStructures.plot import CustomFig
from AudioUtil.DataStructures.note_list import NOTE_FREQS, ClosestNoteFromFreq
from AudioUtil.DataStructures.FFT import FFT
import AudioUtil.DataStructures.output as out
from AudioUtil.MusicAnalyze.util import SignalEnvelope, GetPeakIndices, Normalize, ACF, PitchContour
import copy

def GetHarmonicIndices(peak_indices: list[int], F: np.ndarray, DIFF=40) -> list[float]:
  '''
    peak_indices
      - list of indices of F which correspond to top frequencies
    F
      - list of frequencies
    DIFF
      - acceptable difference between harmonics
  '''

  # FIND THE FREQUENCY W' THE MOST HARMONICS -> F0
  N = len(peak_indices)
  harmonics = []
  # For each item in the array
  for i in range(0, N-1):
    sample_num: int = peak_indices[i]
    F_i = F[sample_num]
    cur_harmonics = []
    # For all subsequet frequencies
    for PI_j in peak_indices[i:N-1]:
      F_j = F[PI_j]
      if (F_j % F_i) - F_i < DIFF :
        cur_harmonics.append(PI_j)  # Found a harmonic
    # -----
    if len(cur_harmonics) > len(harmonics):
      harmonics = cur_harmonics
    if len(harmonics) > N-i:
      break
  #----------------------------------------
  return harmonics


# -----------

def Stretch(pitches: np.ndarray, maxPitchThreshold: int) -> np.ndarray:
    # max_i = np.where(np.isinf(pitches),-np.Inf,pitches).argmax()
    # max = pitches[max_i]
    # print(f'max = {max}')
    stretch: float = maxPitchThreshold / pitches.max()
    print(f'stretch = {stretch}')
    return pitches * stretch

def SlopeSum(y: np.ndarray) -> tuple[np.ndarray]:
    N = y.size
    FollowingSlope = np.zeros((N,))
    NumSameSlopeDirection = np.zeros((N,))

    for i in range(0, N-1):
        sum = y[i]
        j = i+1
        while  ( y[i] > 0 and y[j] > 0 ) or ( y[i] < 0 and y[j] < 0 ):
            sum += y[j]
            j+=1
            if j >= N-1: break
        FollowingSlope[i] = sum
        NumSameSlopeDirection[i] = j - i

    return [FollowingSlope, NumSameSlopeDirection]

def SlopeMean(y: np.ndarray, n=10) -> np.ndarray:
    N = y.size
    R = n//2
    L = n-R
    MEANS = np.zeros((N,))
    for i in range(0, N-1):
        i0 = max(0, i-L)
        i1 = min(N-1, i+R)
        MEANS[i] = np.mean(y[i0:i1])
    return MEANS

def SlopeStdDev(SLOPES: np.ndarray, MEANS: np.ndarray, n=10) -> np.ndarray:
    N = SLOPES.size
    R = n//2
    L = n-R
    DEV = np.zeros((N,))
    # ------------------
    for i in range(0, N-1):
        i0 = max(0, i-L)
        i1 = min(N-1, i+R)
        arr = [ (SLOPES[i] - MEANS[i])**2 for i in range(i0, i1) ]
        DEV[i] = np.sqrt(np.sum(arr)/(n-1))
    # ------------------
    return DEV

def PitchDetectAlg(AS: AudioSignal):

    # params (from signal)
        x = AS.time
        y = AS.signal
        fs = AS.sample_freq

    # params

        maxPitchThreshold = 1000
        frame_length = 4096
        window_size = frame_length // 2
        overlap_size = frame_length // 4
        mean_n = 20
        dev_n = 20
        t = 2

    # PLOTTING

        FIG = CustomFig()

    # 1 - PITCH CONTOUR

        # F0, voiced_flag, voiced_prob = librosa.pyin(y=y, fmin=fmin, fmax=fmax, sr=fs, frame_length=frame_length)
        # F0_times = librosa.times_like(F0)
        F0_times, F0 = PitchContour(y=y, fs=fs, window_size=window_size, overlap_size=overlap_size)
        top_axes = FIG.plot_bottom(x=F0_times, y=F0, label_x='times (s)', label_y='F0 (Estimated)')

    # 2 - STRETCH PITCH CONTOUR

        F0_stretched = Stretch(pitches=F0, maxPitchThreshold=maxPitchThreshold)
        top_axes.plot(F0_times, F0_stretched, color='red')
        main_axes = FIG.plot_bottom(x=F0_times, y=F0_stretched, label_x='times (s)', label_y='F0 (stretched)')

    # 3 - CALCULATE THE SLOPES

        F0_diff: np.ndarray = np.gradient(f=F0)
        main_axes.plot(F0_times, F0_diff, label='Slopes', color='purple')

    # 4 - SUMMING THE SLOPES: WHEN DOES THE SLOPE CHANGE SIGN?

        FollowingSlope, NumSameSlopeDirection = SlopeSum(F0_diff)

    # 5 - SLOPE MEANS

        MEANS = SlopeMean(F0_diff, n=mean_n)
        main_axes.plot(F0_times, MEANS, label='Slope Mean', color='orange')

    # 6 - STD DEV OF SLOPES

        DEVS = SlopeStdDev(F0_diff, MEANS, n=dev_n)
        main_axes.plot(F0_times, DEVS, label='Slope Std Dev',color='green')

    # 7 - ESTIMATE STATUS FOR EACH POINT

        print(f'# means = {MEANS.size}')
        print(f'# std dev = {DEVS.size}')
        print(f'# slopes = {F0_diff.size}')

        NONE = 0
        START_TRANSITION = 1
        END_TRANSITION = 2
        ONSET = 3
        OFFSET = 4

        THRESHOLDS = MEANS + ( DEVS * t )
        main_axes.plot(F0_times, THRESHOLDS, color='red')

        #-----
        N = F0_diff.size
        STATUS = np.zeros((N,))
        FirstTime = True
        i = 0
        while i <  N-1:

            if F0[i] == 0:
                FirstTime = True

            if F0_diff[i] > THRESHOLDS[i]:
                # trajectory change has happened (onset, offset or StartTransition)
                j = int(NumSameSlopeDirection[i])

                if FirstTime :
                    # first trajectory change after a silence ==> movement to reach an 𝑂𝑛𝑠𝑒𝑡
                    STATUS[i] = START_TRANSITION
                    FirstTime = False
                else:
                    # the current point is an 𝑂𝑓𝑓𝑠𝑒𝑡
                    STATUS[i] = OFFSET
                    STATUS[i+1] = START_TRANSITION
                    FirstTime = True

                STATUS[i+j-1] = END_TRANSITION
                STATUS[i+j] = ONSET
                #----
                i += j+1
            else:
                STATUS[i] = NONE
                i += 1

        # ------------------

        # PLOT EACH STATUS
        axes = FIG.plot_bottom(x=x, y=y, label_x='Time(s) + Status', label_y='Amplitude')
        for i in range(0,N-1):
            color = None
            if STATUS[i] == NONE:
                continue
            elif STATUS[i] == ONSET:
                color = 'green'
            elif STATUS[i] == OFFSET:
                color = 'red'
            elif STATUS[i] == START_TRANSITION:
                color = 'orange'
            axes.axvline(x=F0_times[i], color=color)


    # SHOW ALL PLOTS
        main_axes.legend()
        FIG.show()

# ------- custom process


def test():
  AS: AudioSignal = AudioSignal_FromFile('../SampleInput/voiceScale.wav')
  AS.change_dtype(np.float64)
  AS.Normalize()
  PitchDetectAlg(AS)

def testACF():
    AS: AudioSignal = AudioSignal_FromFile('../SampleInput/doh.wav')
    T = AS.time
    fs = AS.sample_freq
    acf = ACF(AS.signal)
    fig = CustomFig()
    ax = fig.plot_bottom(x=T, y=acf)

    min_T = 1 / 900
    max_T = 1 / 100

    ax.set_xlim(min_T, max_T)

    ax.axvline(x=min_T, color='red')
    ax.axvline(x=max_T, color='red')

    s0 = int(min_T * fs)
    s1 = int(max_T * fs)

    print(f's0 = {s0} => t = {T[s0]}, s1 = {s1} => t = {T[s1]}')

    sec = acf[s0:s1]

    max_i = s0 + sec.argmax(axis=0)

    print(f'max_i = {max_i}')

    ax.axvline(x=T[max_i], color='orange')

    fig.show()

def test2():
    AS: AudioSignal = AudioSignal_FromFile('../SampleInput/doh.wav')
    y=AS.signal
    T = AS.time
    fs = AS.sample_freq
    fig = CustomFig()
    ax = fig.plot_bottom(x=T, y=y)

    for i in range(3, 6):
        ws = 512 * (2 ** i)
        F0_times, F0 = PitchContour(y=y, fs=fs, window_size=ws, overlap_size=ws/4)
        ax.plot(F0_times, F0, label=f'ws={ws}')


    ax.legend()
    fig.show()

test2()
