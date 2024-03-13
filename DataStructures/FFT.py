import AudioUtil.DataStructures.plot as plt
import numpy as np
from AudioUtil.DataStructures.AudioSignal import AudioSignal
from scipy.signal import find_peaks
from librosa.util import peak_pick
#---------------------------------------------

class WaveInfo:
  def __init__(self, F: float, AMP: float, PHASE: float) -> None:
    self.F = F
    self.AMP = AMP
    self.PHASE = PHASE

class FFT:
  N: int
  FSTEP: float
  # ----
  initial: np.ndarray
  # ----
  F: np.ndarray
  MAGS: np.ndarray  # i.e, AMPLITUDE
  PHASES: np.ndarray
  # ----
  MIN_F  = 20
  MAX_F = 20000

  def __init__(self, signal: AudioSignal) -> None:
    self.FSTEP = signal.sample_freq / signal.n_samples
    #---
    self.initial = np.fft.fft(signal.signal)
    sig_fft = self.initial / signal.n_samples                   # normalized
    self.MAGS = np.sqrt(sig_fft.real ** 2 + sig_fft.imag ** 2)  # MAGNITUDE only
    self.F = np.linspace(0, (signal.n_samples - 1) * self.FSTEP, signal.n_samples)  # frequency steps
    self.PHASES = np.angle(self.initial) # tan^-1(imag/complex)
    #---
    # Restrict to Frequencies we can actually hear [20, 20000]
    self.RestrictF(self.MIN_F, self.MAX_F)

  def RestrictF(self, min: float, max: float):
    I0 = np.where(self.F >= min)[0][0]
    I1 = np.where(self.F >= max)[0][0]
    self.F = self.F[I0:I1]
    self.MAGS = self.MAGS[I0:I1]
    #---
    self.MIN_F = min
    self.MAX_F = max
    self.N = self.F.size

  def PeakFreqIndices(self, display=False, heightFactor:float=0.03) -> list[float]:
    # Initialize Parameters
      Hz_dist = 10 # in Hz
      sample_dist = Hz_dist / (self.MAX_F-self.MIN_F) * self.N     # Convert distance in Hz to distance in samples
      X = sample_dist/2
      delta = self.MAGS.max() * heightFactor
    # Get the peak frequencies
      peak_indices = peak_pick(
        x=self.MAGS,
        pre_max=X,
        post_max=X,
        pre_avg=X,
        post_avg=X,
        delta=delta,
        wait=sample_dist
      )

    # Display/Plot the results
      if(display):
        print(f'PEAK FREQUENCIES: {len(peak_indices)}:')
        print(f"\t{'Freq (Hz)':<15}MAG")
        for i in peak_indices:
          F = self.F[i]
          MAG = self.MAGS[i]
          print(f'\t{F:<15}{MAG}')

        return peak_indices

  def InverseFFT(self):
    return np.fft.ifft(self.initial)
