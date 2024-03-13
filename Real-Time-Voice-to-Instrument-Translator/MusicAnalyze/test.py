import AudioUtil.DataStructures.plot as plt
import numpy as np
from AudioUtil.DataStructures.AudioSignal import AudioSignal
from AudioUtil.DataStructures.note_list import notes
import AudioUtil.DataStructures.output as out
#---
from AudioUtil.MusicAnalyze.OnsetDetection import get_onset_indices
from AudioUtil.MusicAnalyze.RecreateNote import GetSignalFromOnset

# -----------------------------------------------------------------------------

# 1 - GET THE SAMPLE FROM A FILE
sig = AudioSignal()
sig.INITfromFile('../SampleInput/sample.wav')
sig.change_dtype(np.float64)

# 2 - GET ONSET TIMES
onset_indices = get_onset_indices(sig)

# FOR EACH TIME INTERVAL, RECREATE IT AS A DIGITAL SIGNAL. ADD IT TO A TOTAL
output_signals = []
i = 0
N = sig.time.size
for index in onset_indices:
  start_i = index
  end_i = N-1 if i >= onset_indices.size -1 else onset_indices[i+1]
  print(f'frame {i} = [{start_i},{end_i}]')

  # Convert that interval into a digital signal
  slice = AudioSignal(sig.n_channels, sig.sample_freq, sig.n_samples, sig.time, sig.signal)
  slice.slice(start_i, end_i)

  plot = plt.CustomFig('SLICE')
  plot.AddPlot(slice)
  plot.Show()

  result = GetSignalFromOnset(slice)

  #Add that to the output signal:
  padded_result = np.pad(result.signal, (start_i, N-end_i), 'constant')    # pad leading and trailing zeroes
  output = AudioSignal(sig.n_channels, sig.sample_freq, N, sig.time, padded_result)

  output_signals.append(output)

  i+=1

final_result = out.combine_signals(*output_signals)


final_result_plot = plt.CustomFig('Final Result')
final_result_plot.AddPlot(final_result)
final_result_plot.Show()