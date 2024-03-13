import numpy as np
from scipy import signal as sg
import math
from AudioUtil.DataStructures.plot import CustomFig
from numpy import ndarray
from AudioUtil.DataStructures.AudioSignal import AudioSignal

# linspace: returns evenly spaced values within a given interval. num no. of elements are returned.
def LinearToDecibel(value: float) -> float :
  return 20.0 * math.log10(value)

def DecibelToLinear(db_val: float) -> float :
  return 10 ** (db_val / 20)

def TimeArray(duration: float, fs: float = 44100) -> ndarray:
  # Note, we take the floor for # steps (so our time will always be less)
  num_steps = int(np.floor(duration * fs))
  return np.linspace(start=0, stop=duration, num=num_steps, endpoint=False)


def create_pure_tone(freq: float, duration: float, peak:int, fs: float = 44100)->AudioSignal:
  # pure tone = single sine wave, at a fixed magnitude (peak) & frequency (freq)
  T = TimeArray(duration, fs)
  signal = np.sin(2*np.pi * freq * T) * peak
  return AudioSignal(1, fs, T.size, T, signal)

def cos_pure_tone(freq, duration:float, peak:int, fs:float=44100, phase_shift=0)->AudioSignal:
  T = TimeArray(duration, fs)
  signal = np.cos(2*np.pi * freq * T + phase_shift) * peak
  return AudioSignal(1, fs, T.size, T, signal)

def create_triangle_wave(freq, duration, peak, fs):
  T = TimeArray(duration, fs)
  return sg.sawtooth(2*np.pi*freq*T, width=0.5) * peak

def create_square_wave(freq, duration, peak, fs):
  T = TimeArray(duration, fs)
  return sg.square(2*np.pi*freq*T) * peak

def combine_signals(*sigs: AudioSignal, showFig=False):
  # Add all the signals together
  max_samples = 0
  signals = []

  # plot it:
  if(showFig): fig = CustomFig('Combined Signals')

  for sig in sigs:
    n = sig.n_samples
    if n > max_samples: max_samples = n
    else:
      extra = max_samples - n + 1
      np.pad(sig, (0, extra))

    signals.append(sig.signal)
    if(showFig): fig.AddPlot(sig)

  combined_signal = sum(signals)

  if(showFig): fig.Show()

  return AudioSignal(sigs[0].n_channels, sigs[0].sample_freq, max_samples, sigs[0].time, combined_signal)


def Amplify(signal, dBs: int):
  # amplify a signal by a specified number of dBs
  amount = DecibelToLinear(dBs)
  factor = 1
  if dBs < 0: factor = -1
  return signal * (1+amount*factor)
