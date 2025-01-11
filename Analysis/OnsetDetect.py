import numpy as np
import copy
import librosa
import DataStructures.plot as plt
from  DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from  DataStructures.plot import CustomFig
from  Analysis.util import EmptyArr, sec2samp, Signal_Mean_StdDev
from scipy.signal import savgol_filter # filter out the noise
from  Analysis.YIN import yin
from  Analysis.SilenceDetect import DetectSilence

NONE = 0
START_TRANSITION = 1
END_TRANSITION = 2
ONSET = 3
OFFSET = 4

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

def ToNearestNotes(PITCH: np.ndarray) -> np.ndarray:
    out = PITCH[:]
    i = 0
    for pitch in PITCH:
        try:
            note = librosa.hz_to_note(int(round(pitch)))
            note_hz = librosa.note_to_hz(note)
        except:
            note_hz = 0
        out[i] = note_hz
        i+=1
    return out

class PointInfo:
    def __init__(self, pitch: float, cnt: int):
        self.pitch = pitch
        self.cnt = cnt
        self.long = False

def RemoveQuickPitchChange(PITCH: np.ndarray, PITCH_T: np.ndarray, dT=0.1):
    '''
        We could have short deviations within short deviations
        maintain a stack
        top of th stack:
        - current value we are on
        - cnt of the value
        - from-value
    '''

    N = PITCH.size
    fs = PITCH.size / PITCH_T[-1]
    dS = sec2samp(dT, fs)

    stack: list[PointInfo] = []

    for i in range(N):
        p = PITCH[i]
        if len(stack) == 0:
            print(f'stack.len == 0')
            stack.append(PointInfo(p, 1))
            continue
        # <- item on the stack
        top = stack[-1]
        if p == top.pitch:
            top.cnt += 1
            if not top.long and top.cnt >= dS:
                # <- long enough to keep
                top.long = True
                print(f'top.long <- true')
        else:
            # <- might not be a <last> item
            if len(stack) == 1:
                # <- first new pitch discovered
                print(f'stack.len == 1')
                stack.append(PointInfo(p, 1))
            else:
                last = stack[-2]
                if p == last.pitch:
                    # <- deviated from the last pitch but came back
                    last.cnt += 1
                    if not top.long:
                        # <- too short to keep
                        print(f'too short:')
                        print(f'\t last.pitch = {last.pitch}')
                        print(f'\t pitch = {top.pitch}')
                        print(f'\t [{PITCH_T[i-top.cnt-1]}, {PITCH_T[i]}]')
                        last.cnt += top.cnt
                        stack.pop()
                        del top
                else:
                    # <- p is a (possibly) unseen pitch (need to search down stack)
                    stack.append(PointInfo(p, 1))

    # i = 0
    # start = 0
    # for PI in stack:
    #     print(f'stack[{i}]:')
    #     print(f'\t pitch = {PI.pitch}')
    #     print(f'\t [{PITCH_T[start]}, {PITCH_T[start+PI.cnt]}]')
    #     start += (PI.cnt + 1)
    #     i+=1


    NEW_PITCH = EmptyArr(N)
    i = 0
    for PI in stack:
        NEW_PITCH[i:i+PI.cnt] = PI.pitch

    # print(NEW_PITCH)

    return NEW_PITCH

    # P = PITCH[:]
    # N = PITCH.size
    # fs = PITCH.size / PITCH_T[-1]
    # dS = sec2samp(dT, fs)
    # p = PITCH[0]
    # cnt = 0
    # first_dev = None
    # candidates = []     # stack
    # for i in range(N):
    #     # print(f'PITCH[i] = {PITCH[i]}')
    #     if PITCH[i] != p:
    #         if cnt == 0:
    #             #print(f'\t NEW dev from {p}] @ i = {i}')
    #             first_dev = i
    #             cnt += 1
    #         elif cnt >= dS:
    #             # this is more than a short deviation

    #             # we don't know if this is the p we want tho (ie: it could be in a short deviation)

    #             p = PITCH[i]
    #             #print(f'\t more than a short deviation ... NEW p: {p}]')
    #             cnt = 0
    #             first_dev = None
    #         else:
    #             cnt += 1
    #     else:
    #         # it's the pitch we want
    #         if cnt != 0:
    #             # get rid of the short deviation
    #             P[first_dev:i] = p
    #             cnt = 0
    #             first_dev = None
    return P


# -----------
MAX = None
def Stretch(pitches: np.ndarray, maxPitch: int) -> np.ndarray:
        '''
            To counteract any adverse effect of this wide pitch frequency range on the slopes,
            the F0s are stretched to be on the almost same pitch frequency range: [min_f, max_f]
        '''

    # METHOD 1

        N = pitches.size

        # CALCULATE MAX (up until i) for each i

        global MAX
        MAX = np.zeros(N)

        cur_max = pitches[0]

        for i in range(N):
            MAX[i] = cur_max
            if pitches[i] > cur_max:
                cur_max = pitches[i]

        # STRETCH THE PITCH CURVE USING MAX AT EACH POINT

        stretched = copy.deepcopy(pitches)

        for i in range(N):
            m = MAX[i]
            if m != 0:
                stretched[i] = pitches[i] * ( maxPitch / m )

        return stretched

def FollowingSlopes(pitchSlopes: np.ndarray) -> tuple[np.ndarray]:
    '''
        INPUT: pitchSlopes <ndarray>
        OUTPUT: FollowingPitchSlope <ndarray>
        - at point i -> sum all the slopes ahead of i until no longer the same slope
    '''
    N = pitchSlopes.size
    FollowingPitchSlope = EmptyArr(N)
    NumSameSlopeDirection = EmptyArr(N)
    #-----
    for i in range(0, N-1):
        sumSlope = pitchSlopes[i]
        sameSignSlopes = [ sumSlope ]
        numSame = [ 0 ]
        j = i+1
        while j < N and not abs(pitchSlopes[j]) < 1 and ( pitchSlopes[i] * pitchSlopes[j] )  > 0:
            # <- same signed slope
            sumSlope += pitchSlopes[j]
            sameSignSlopes.insert(0, sumSlope)      # add to front
            numSame.insert(0, j-i)                # add to front
            j+=1
        #-------
        # [i:j] = subarray where all are the same sign
        FollowingPitchSlope[i:j] = sameSignSlopes
        NumSameSlopeDirection[i:j] = numSame
        i += j
    #-----
    return [FollowingPitchSlope, NumSameSlopeDirection]

def GetStatus(SLOPES: np.ndarray, FollowingPitchSlope: np.ndarray, NumSameSlopeDirection: np.ndarray, THRESHOLDS: np.ndarray) -> np.ndarray:
    N = FollowingPitchSlope.size
    STATUS = np.zeros((N,))
    #-------
    OnsetSeen = False
    i = 0
    while i <  N-1:
        if np.abs(FollowingPitchSlope[i]) > np.abs(THRESHOLDS[i]) and SLOPES[i] > 10:
            # trajectory change has happened (onset, offset or StartTransition)
            j = int(NumSameSlopeDirection[i])

            if not OnsetSeen:
                # first trajectory change after a silence ==> movement to reach an 𝑂𝑛𝑠𝑒𝑡
                STATUS[i] = START_TRANSITION
            else:
                # the current point is an 𝑂𝑓𝑓𝑠𝑒𝑡
                STATUS[i] = OFFSET
                STATUS[i+1] = START_TRANSITION
                OnsetSeen = False
            #----
            STATUS[i+j-1] = END_TRANSITION
            STATUS[i+j] = ONSET
            OnsetSeen = True
            #----
            i += j+1
        else:
            i += 1


    return STATUS

def PitchContour(y: np.ndarray, x: np.ndarray, fs, fmin, fmax) -> tuple[np.ndarray]:
    '''
        PARAMETERS
        ----------
        frame_length :  length of the frames (in samples)
        win_length :    length of the window for calculating autocorrelation (in samples)
        hop_length :    number of audio samples between adjacent YIN predictions
    '''
    frame_length = 4096
    window_size = frame_length // 2
    hop_length = frame_length // 4

    PITCH = yin(
        y=y,
        fmin=fmin,
        fmax=fmax,
        sr=fs,
        frame_length=frame_length,
        win_length=window_size,
        hop_length=hop_length,
    )


    # PITCH, voiced_flag, voiced_prob = librosa.pyin(
    #     y=y,
    #     fmin=fmin,
    #     fmax=fmax,
    #     sr=fs,
    #     frame_length=frame_length,
    #     win_length=window_size,
    #     hop_length=hop_length
    # )

    # Calculate the Times for each frame

    n_frames = PITCH.size
    PITCH_T = np.zeros((n_frames))
    for f_num in range(n_frames):
        PITCH_T[f_num] = x[hop_length * f_num]

    return [PITCH_T, PITCH]

def pitch_and_onset_detection(AS: AudioSignal, plotIt=False):

    # from signal
        x = AS.time
        y = AS.signal
        fs = AS.sample_freq

    # params

        fmin=65
        fmax=900
        n = 30
        t = 2.5
        min_note_dist = 0.15 # *** can't be too much (breaks for 0.2)
        silence_threshold = 0.005

    # 0 - PRE PROCESSING

        y = savgol_filter(y, 512, 2)

        y = DetectSilence(t=t, signal=y, threshold=silence_threshold)

    # 1 - PITCH CONTOUR

        PITCH_T, PITCH = PitchContour(y, x, fs, fmin=fmin, fmax=fmax)

        PITCH = ToNearestNotes(PITCH)

        N = PITCH.size

    # 2 - STRETCH PITCH CONTOUR

        STRETCH_PITCH =  Stretch(pitches=PITCH, maxPitch=fmax)

    # 3 - CALCULATE THE SLOPES

        SLOPES: np.ndarray = np.gradient(PITCH, PITCH_T)

    # 4 - SUMMING THE SLOPES: WHEN DOES THE SLOPE CHANGE SIGN?

        FollowingPitchSlope, NumSameSlopeDirection = FollowingSlopes(SLOPES)

    # 5 - MEAN and STD-DEV of SLOPES

        MEANS, DEVS = Signal_Mean_StdDev(FollowingPitchSlope, n=n)

    # 7 - ESTIMATE STATUS FOR EACH POINT

        THRESHOLDS = MEANS + ( DEVS * t )

        STATUS = GetStatus(SLOPES, FollowingPitchSlope, NumSameSlopeDirection, THRESHOLDS)

        if plotIt:

            FIG = CustomFig()

            axes = FIG.plot_bottom(x=x, y=y, label_x='Time(s)', label_y='Amplitude')

            pitch_axes = FIG.plot_bottom(x=PITCH_T, y=PITCH, label_x='Time(s)', label_y='F0')
            pitch_axes.plot(PITCH_T, STRETCH_PITCH, label='Stretched Pitch', color='orange')
            # pitch_axes.axhline(y=fmin, color='black')
            # pitch_axes.axhline(y=fmax, color='black')

            for i in range(0,N-1):
                color = None
                t = PITCH_T[i]
                if STATUS[i] == NONE:
                    continue
                elif STATUS[i] == ONSET:
                    color = 'orange'
                elif STATUS[i] == OFFSET:
                    color = 'red'
                elif STATUS[i] == START_TRANSITION:
                    color = 'green'
                elif STATUS[i] == END_TRANSITION:
                    color = 'purple'

                axes.axvline(x=t, color=color)

            # FIG.CustomLegend(axes, [
            #     ['orange', 'ONSET'],
            #     ['red', 'OFFSET'],
            #     ['green', 'Start Transition'],
            #     ['purple', 'End Transition']
            # ])

            main_axes = FIG.plot_bottom(x=PITCH_T, y=STRETCH_PITCH, label_x='times (s)', label_y='F0 (stretched)')
            main_axes.set_xlim(0, PITCH_T[-1])
            main_axes.plot(PITCH_T, SLOPES, label='Slopes', color='purple')
            main_axes.plot(PITCH_T, np.abs(FollowingPitchSlope), label='Following Slope', color='grey')
            main_axes.plot(PITCH_T, NumSameSlopeDirection, label='Num Same Slope', color='black')
            main_axes.plot(PITCH_T, MEANS, label='Slope Mean', color='orange')
            main_axes.plot(PITCH_T, DEVS, label='Slope Std Dev', color='green')
            main_axes.plot(PITCH_T, THRESHOLDS, label='Threshold', color='red')
            main_axes.legend()

        # SHOW ALL PLOTS

            FIG.show()


    # 8 - RETURN NEEDED INFORMATION

        return [PITCH_T, PITCH, STATUS]

# ------- TESTS -------
name = 'voiceScale_Filtered'
AS: AudioSignal = AudioSignal_FromFile(f'../SampleInput/{name}.wav')
y: np.ndarray = AS.signal
t = AS.time
fs = AS.sample_freq
pitch_and_onset_detection(AS, plotIt=True)