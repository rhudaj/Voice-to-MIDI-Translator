import AudioUtil.DataStructures.plot as plt
from AudioUtil.DataStructures.AudioSignal import AudioSignal
import AudioUtil.DataStructures.output as out
#-----
import librosa
import numpy as np

# https://musicinformationretrieval.com/onset_detection.html

def get_onset_indices(sig: AudioSignal):

  '''
  A sample n is selected as an peak if the corresponding x[n] fulfills the following three conditions:

    x[n] == max(x[n - pre_max:n + post_max])

    x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta

    n - previous_n > wait

  where previous_n is the last sample picked as a peak (greedily).
  '''

  onset_samples = librosa.onset.onset_detect(
    y=sig.signal,
    sr=sig.sample_freq,
    units='samples',         # return indicies
    backtrack=True,         # If True, detected onset events are backtracked to the nearest preceding minimum of energy.
    normalize=True,
    pre_max=50,
    post_max = 50
  )

  print(onset_samples)

  return onset_samples
