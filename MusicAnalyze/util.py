import math
import numpy as np
import librosa

def LinearToDecibel(value: float) -> float :
  return 20.0 * math.log10(value)

def DecibelToLinear(db_val: float) -> float :
  return 10 ** (db_val / 20)

def sec2samp(s: float, fs: float) -> int:
  return  int(s * fs)

def Normalize(y: np.ndarray):
  return y / y.max()

def SignalEnvelope(x: np.ndarray, y: np.ndarray, win_len_samps: int) -> np.ndarray:
  # setup params
    y = np.abs(y)
  # get the moving average of the signal
    n = win_len_samps
    y = np.cumsum(y, dtype=float)
    y[n:] = y[n:] - y[:-n]
    y = y / n
    return y

def GetPeakIndices(
  y: np.ndarray,
  max_range: int,
  avg_range: int,
  delta: float = 0,
  wait: float = 0
):
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

# -------------------- STD MATH FUNCTIONS --------------------

def Variance(values: np.ndarray, avg: float, sample: bool = False) -> float:
	var = 0
	N = values.size
	for val in values:
		var += 2 ** (val - avg) #*** avg over all values
	if sample:
		var /= (N -1)
	else:
		var /= N
	return var

def StdDeviation(values: np.ndarray, avg: float, sample: bool = False) -> float:
  var = Variance(values, avg, sample)
  sd = math.sqrt(var)
  return sd

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

def PitchContour(
    y: np.ndarray,
    fs: int,
    window_size: int,
    overlap_size: int,
    fmax = 900,
    fmin = 70
  ) -> tuple[np.ndarray]:
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