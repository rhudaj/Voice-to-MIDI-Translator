# -------------------- IMPORTS --------------------
# ----- EXTERNAL
import math
import numpy as np
from numpy import ndarray
import librosa
from scipy import signal as sg
# ----- INTERNAL
from AudioUtil.DataStructures.plot import CustomFig
from scipy.signal import hilbert
# -------------------- Creating / Modifying (Audio Only) Signals --------------------

def TimeArray(duration: float, fs: float = 44100) -> ndarray:
  # Note, we take the floor for # steps (so our time will always be less)
  num_steps = int(np.floor(duration * fs))
  return np.linspace(start=0, stop=duration, num=num_steps, endpoint=False)

def create_pure_tone(freq: float, duration: float, peak:int, fs: float = 44100, phase_shift=0)->tuple[np.ndarray]:
  # pure tone = single sine wave, at a fixed magnitude (peak) & frequency (freq)
  T = TimeArray(duration, fs)
  signal = np.sin((2*np.pi * freq * T) + phase_shift) * peak
  return [T, signal]

def cos_pure_tone(freq, duration:float, peak:int, fs:float=44100, phase_shift=0)->tuple[np.ndarray]:
  T = TimeArray(duration, fs)
  signal = np.cos(2*np.pi * freq * T + phase_shift) * peak
  return

def create_triangle_wave(freq, duration, peak, fs):
  T = TimeArray(duration, fs)
  return sg.sawtooth(2*np.pi*freq*T, width=0.5) * peak

def create_square_wave(freq, duration, peak, fs):
  T = TimeArray(duration, fs)
  return sg.square(2*np.pi*freq*T) * peak

def combine_signals(*sigs: np.ndarray, fs) -> tuple[np.ndarray]:
  # Add all the signals together
  signals = []
  N = 0
  for sig in sigs:
    n = sig.size
    if n > N:
      N = n
    else:
      extra = N - n + 1
      np.pad(sig, (0, extra))

    signals.append(sig)

  combined_signal = sum(signals)

  T = np.arange(start=0, stop=N-1, step=N/fs)

  return [T, combined_signal]

def Amplify(signal, dBs: int):
  # amplify a signal by a specified number of dBs
  amount = DecibelToLinear(dBs)
  factor = 1
  if dBs < 0: factor = -1
  return signal * (1+amount*factor)

def tones2signal(freqs: list[float], phases: list[float], amps: list[float], duration: float, fs):
  signals = []
  i = 0
  for F in freqs:
    tone = create_pure_tone(freq=F, duration=duration, peak=amps[i], fs=fs, phase_shift=phases[i])
    signals.append(tone)
    i+=1
  sum = combine_signals(*signals)
  return sum

# -------------------- HELPER FUNCTIONS --------------------

def EmptyArr(N: int) -> np.ndarray:
  return np.zeros((N,))

def LinearToDecibel(value: float) -> float :
  return 20.0 * math.log10(value)

def DecibelToLinear(db_val: float) -> float :
  return 10 ** (db_val / 20)

def sec2samp(s: float, fs: float) -> int:
  return  int(s * fs)

def Normalize(y: np.ndarray):
  return y / y.max()


def Envelope(y: np.ndarray, w = 4096) -> np.ndarray:
  analytic_signal = hilbert(y)
  return np.abs(analytic_signal)

def GetPeakIndices(y: np.ndarray, max_range: int, avg_range: int, delta: float = 0, wait: float = 0):
  max_range //= 2
  avg_range //= 2
  peak_indices = librosa.util.peak_pick(
    x=y,
    pre_max=max_range,
    post_max=max_range,
    pre_avg=avg_range,
    post_avg=avg_range,
    delta=delta,
    wait=wait
  )
  return peak_indices

def RemoveDC(y: np.ndarray):
  '''
    essentially, make mean = 0
  '''
  y = y - np.mean(y)

def mu_law(x, mu=255):
  # map each input value to a value
  return np.sign(x) * np.log(1+mu*np.abs(x)) / np.log(1+mu)


# -------------------- STD MATH FUNCTIONS --------------------

def Variance(values: np.ndarray, avg: float, sample: bool = False) -> float:
  N = values.size
  #-------------
  var = 0
  for val in values:
    var += (val - avg) ** 2 #*** avg over all values
  if sample:
    var /= (N -1)
  else:
    var /= N
  return var

def StdDeviation(values: np.ndarray, avg: float, sample: bool = False) -> float:
  return math.sqrt(Variance(values, avg, sample))

def Signal_Mean_StdDev(y: np.ndarray, n=10) -> tuple[np.ndarray]:
    N = y.size
    noBefore = n // 2
    noAfter = n - noBefore
    #------
    MEANS = EmptyArr(N)
    DEVS = EmptyArr(N)
    #------
    for i in range(N):
        s0 = max(0, i - noBefore)
        s1 = min(N-1, i + noAfter + 1)
        values: np.ndarray = y[s0 : s1]
        MEANS[i] = np.mean(values)
        DEVS[i] = StdDeviation(values, MEANS[i], True)
    #------
    return [MEANS, DEVS]


# -------------------- PITCH CONTOUR --------------------

def ACF(y: np.ndarray) -> np.ndarray:
  """
    The autocorrelation is the Inverse FFT of the FFT of the signal,
    multiplied by the complex conjugate of the FFT of the signal.
    The power spectrum of a signal is the exact same thing as the FFT of its autocorrelation!
  """
  N = y.size
  pow2 = int(2 ** np.ceil(np.log2(N)))
  Y = np.zeros(pow2,float)
  Y[:N] = y
  FFT = np.fft.fft(y)
  CONJ = np.conjugate(FFT)
  acf = np.fft.ifft( FFT * CONJ ).real / float(N)
  acf /= acf[0]      # normalize
  return  acf[:N]

def ACFmax_index(acf: np.ndarray, s0, s1):
  try:  x = np.where(acf < 0)[0][0]
  except: x = 0
  s0 = max(s0, x)
  sec = acf[s0:s1]
  return sec.argmax(axis=0)

def Pitch_Contour(y: np.ndarray, fs: int, window_size: int, overlap_size: int, fmax = 900, fmin = 70) -> tuple[np.ndarray]:
  #-------
  stepSize = int(window_size - overlap_size)
  N = int( ((y.size - window_size) // stepSize) + 1)
  #--------
  F0 = np.zeros((N,))
  TIMES = np.zeros((N,))
  #--------
  min_T = 1 / fmax
  max_T = 1 / fmin
  s0 = int(min_T * fs)
  s1 = int(max_T * fs)
  #-------
  t0 = 0
  for i in range(0, N):
    # Get the window
    t0 = i * stepSize  # index to start
    t1 = t0 + window_size
    window = y[t0:t1]
    #-------
    acf = ACF(window)
    #-------
    max_i = ACFmax_index(acf, s0, s1)
    T = min_T + ( max_i / fs )
    #-------
    F0[i] = 1/T
    TIMES[i] = t0 / fs
    #-------
    t0+=stepSize
  #--------
  return [TIMES, F0]


# -------------------- COMPRESSION --------------------
