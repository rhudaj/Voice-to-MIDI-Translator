import numpy as np
import librosa
import copy
import AudioUtil.DataStructures.plot as plt
import matplotlib.pyplot as pyplot
from matplotlib.axes import Axes
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.DataStructures.note_list import NOTE_FREQS
import AudioUtil.DataStructures.output as out
from AudioUtil.MusicAnalyze.util import sec2samp
#------
from AudioUtil.MusicAnalyze.NoteTimes.OnsetDetect import get_onset_indices
from AudioUtil.MusicAnalyze.NoteTimes.DetectSilence import SilenceIntervals


def GetNoteIntervals(AS: AudioSignal):

  AS: AudioSignal = AudioSignal_FromFile('../SampleInput/voiceScale.wav')
  AS.change_dtype(np.float64)
  fs = AS.sample_freq

  # Get a KNOWN sample of silence
  silence = copy.deepcopy(AS)
  silence.slice(sec2samp(13,fs), sec2samp(15,fs))
  silence.output_wav('silence')

  # Based on this sample of silence, identify the intervals of silence
  silent_intervals = SilenceIntervals(AS, silence)

  # Get the onset indices
  onset_indices = get_onset_indices(AS)

  # PLOTTING

  axes: Axes = pyplot.subplot()

  axes.plot(AS.time, AS.signal)

  for OI in onset_indices:
    axes.axvline(x=OI/fs, color='black')

  for SI in silent_intervals:
    axes.axvline(x=SI[0]/fs, color = 'red')
    axes.axvline(x=SI[1]/fs, color = 'orange')




def test():
  AS: AudioSignal = AudioSignal_FromFile('../SampleInput/voiceScale.wav')
  AS.change_dtype(np.float64)
  fs = AS.sample_freq
  GetNoteIntervals(AS)

  pyplot.show()


test()
