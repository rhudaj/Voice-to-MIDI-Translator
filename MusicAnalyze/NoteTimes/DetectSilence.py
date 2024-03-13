import numpy as np
import librosa
import AudioUtil.DataStructures.plot as plt
import matplotlib.pyplot as pyplot
from matplotlib.axes._axes import Axes
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.DataStructures.FFT import FFT
import AudioUtil.DataStructures.output as out
from AudioUtil.MusicAnalyze.util import sec2samp
from AudioUtil.MusicAnalyze.NoteTimes.OnsetDetect import get_onset_indices
import copy

'''
  IDENTIFYING NOTE ONSETS USING ONLY SOUND LEVELS (NO Frequency Analysis)
'''

plot = False

if plot:
  fig, ax = pyplot.subplots(nrows=1, sharex=True)
  ax: Axes = ax

# -----------------------------------

def SilenceIntervals(AS: AudioSignal, silence: AudioSignal, plot = False) -> list[tuple]:
  # 1 - Find the threshold (what is no longer silence?
  envelope = silence.SignalEnvelope()
  FACTOR = 2  # how much more past the mean are we willing to go to?
  threshold = np.mean(envelope) * FACTOR
  # 2 - Identify when silence ends and when it starts
  envelope = AS.SignalEnvelope()

  above = np.where(envelope <= threshold)[0]  # indicies where the threshold is below

  starts = []
  ends = []

  N = envelope.size
  FS = AS.sample_freq
  WAIT = sec2samp(0.1, FS)
  above = False
  last_any = - WAIT
  for i in range(0, N-1):
    if above == False and envelope[i] > threshold:  # new potential starting point
      if i - last_any > WAIT:  # has it been long enough since the last 'above' point
        starts.append(i)
        above = True
        last_any = i
    elif above == True and envelope[i] < threshold:
      if i - last_any > WAIT: # has it been long enough since the last 'below' point
        ends.append(i)
        above = False
        last_any = i

  # Get the intervals where silence occurs
  intervals = []

  I0 = [0, starts[0]]
  intervals.append(I0)

  num = 0
  for end_i in ends:
    I = [end_i, starts[num+1]]
    print(f'Interval {num} : {I}')
    intervals.append(I)
    num += 1

  # PLOT

  if plot :

    ax.plot(AS.time, AS.signal)

    ax.axhline(y=threshold, color='black')

    for i in starts:
      ax.axvline(AS.time[i], color='red')

    for i in ends:
      ax.axvline(AS.time[i], color='orange')

    pyplot.show()

  return intervals

# -------------- TESTING --------------

def test():
  AS: AudioSignal = AudioSignal_FromFile('../SampleInput/voiceScale.wav')
  AS.change_dtype(np.float64)
  fs = AS.sample_freq

  # Get a KNOWN sample of silence
  silence = copy.deepcopy(AS)
  silence.slice(sec2samp(13,fs), sec2samp(15,fs))
  silence.output_wav('silence')

  # Based on this sample of silence, identify the intervals of silence
  silent_intervals = SilenceIntervals(AS, silence)

  # Remove these intervals
  for I in silent_intervals:
    AS.signal[I[0]:I[1]] = 0

  if plot:
    plot = plt.CustomFig('Silence Removed', AS)
    pyplot.show()