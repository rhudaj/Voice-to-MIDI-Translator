import AudioUtil.DataStructures.plot as plt
import numpy as np
from AudioUtil.DataStructures.AudioSignal import AudioSignal
from AudioUtil.DataStructures.note_list import notes
from scipy.signal import find_peaks
import AudioUtil.DataStructures.output as out

'''
THIS FILE IS FOR:
  GIVEN A SINGLE NOTE AS INPUT,
  RECREATE THAT NOTE FROM SCRATCH, AS A DIGITAL SIGNAL (SUM OF SINUSOIDS)
'''

# STEP 1 : GET THE FFT

def GetFFT(signal: AudioSignal):
  #---
  Fs = signal.sample_freq
  N = signal.n_samples
  fstep = Fs/N
  #---
  sig_fft: np.ndarray = np.fft.fft(signal.signal) / N   # normalized ***
  F = np.linspace(0, (N - 1) * fstep, N)  # frequency steps
  #---
  # Restrict to Frequencies we can actually hear [20, 20000]
  sig_fft = sig_fft[20:20000]
  F = F[20:20000]
  #---
  # Get the Magnitudes for each frequency:
  mags = np.sqrt(sig_fft.real ** 2 + sig_fft.imag ** 2)
  #---
  return [F, mags]

# STEP 2 : FIND PEAK FREQUENCIES FROM THE FFT

def GetPeakFreqs(x: np.ndarray, y: np.ndarray, plot: plt.CustomFig=None):
  # Initialize Parameters
    min_height = y.max() * 0.5
    Hz_dist = 20 # in Hz
    # Convert distance in Hz to distance in samples
    sample_dist = (Hz_dist / (x.max()-x.min())) * x.size

    peak_indices, properties = find_peaks(y, height=min_height, distance=sample_dist)

  # Display the results

    peak_freqs = []

    print('GetPeaks:')
    print(f"\t{'freq (Hz)':<15}Peak Value")
    for i in peak_indices:
      print(f'\t{x[i]:<15}{y[i]}')
      if(plot): plot.addPoint(x[i], y[i])
      peak_freqs.append(x[i])

    return peak_freqs

# STEP 3 : COMBINE THOSE FREQUENCIES TO A SINGLE SIGNAL

def CombineFreqs2Signal(peak_freqs: list[float], duration: float):
  print(f'CombineFreqs2Signal, peak_freqs.size = {len(peak_freqs)}')
  signals = []
  for freq in peak_freqs:
    signal = out.create_pure_tone(freq, duration, 1)
    signals.append(signal)

  return out.combine_signals(*signals)

# STEPS 1, 2 & 3 COMBINED

def GetSignalFromOnset(signal: AudioSignal):
  # where signal is a short section with 1 note played
  F, fft = GetFFT(signal)
  peak_freqs = GetPeakFreqs(x=F, y=fft)
  sum = CombineFreqs2Signal(peak_freqs, signal.duration)

  return sum