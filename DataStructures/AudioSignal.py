import numpy as np
from scipy.io.wavfile import write, read
from AudioUtil.MusicAnalyze.util import DecibelToLinear, sec2samp
import librosa

class AudioSignal:
  def __init__(
    self,
    n_channels,
    sample_freq,
    n_samples,
    time: np.ndarray,
    signal: np.ndarray
  ) -> None:
    # create object from parameters
    self.n_channels = n_channels
    self.sample_freq = sample_freq
    self.n_samples = n_samples
    self.duration = n_samples / sample_freq
    self.time = time
    self.signal = signal
    self.dtype = signal.dtype

    self.change_dtype(np.float64)
    self.Normalize()

  def output_wav(self, name: str):
    write(filename=f'{name}.wav', rate=self.sample_freq, data=self.signal)

  def Normalize(self):
    abs_val = np.abs(self.signal)
    max = abs_val.max()
    min = abs_val.min()
    print(f'Normalizing \n\t max = {max} @ t = {self.time[self.signal.argmax()]}')
    self.signal = (self.signal - min) / (max - min)

  def get_derivative(self):
    dx = 1 / self.sample_freq
    derivative = self # copy
    derivative.signal = np.diff(self.signal, prepend=self.signal[0])  #prepend value so that y does not lose size by 1
    return derivative

  def slice(self, i_start: int, i_end: int):
    start_t = self.time[i_start]
    end_t = self.time[i_end]
    self.time = self.time[i_start:i_end]
    self.signal = self.signal[i_start:i_end]
    self.duration = end_t - start_t
    self.n_samples = i_end - i_start
    self.time -= start_t

  def change_dtype(self, dtype):
    if(self.dtype == dtype): return

    old_peak = np.iinfo(self.dtype).max if np.issubdtype(self.dtype, np.integer) else 1
    new_peak = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    factor = min(old_peak, new_peak) / max(old_peak, new_peak)

    self.signal = (self.signal * factor).astype(dtype) # convert

    print(f'Changed dtype: ${self.dtype} --> ${dtype}')
    self.dtype = dtype

  def FrequencyFilter(self, F_START: float, F_END: float):
    # get FFT & corresponding frequencies
    fft = np.fft.fft(self.signal)
    FSTEP = self.sample_freq / self.n_samples
    FREQS = np.linspace(0, (self.n_samples - 1) * FSTEP, self.n_samples)  # frequency steps

    # set low freqs to zero
    i = np.where(FREQS >= F_START)[0][0]
    fft[0:i] = np.zeros(i, dtype=np.complex_) # array of i zeroes

    # set high freqs to zero
    i = np.where(FREQS >= F_END)[0][0]
    filter = np.zeros(FREQS.size - i - 1, dtype=np.complex_)
    fft[i:fft.size-1] = filter

    # inverse the fft to get OG signal
    filtered_signal = np.fft.ifft(fft).real

    # RETURN
    self.signal = filtered_signal
    return self

  def FrequencyGain(self, F_START: float, F_END: float, gainDB: float):
    # get FFT & corresponding frequencies
    fft = np.fft.fft(self.signal)
    FSTEP = self.sample_freq / self.n_samples
    FREQS = np.linspace(0, (self.n_samples - 1) * FSTEP, self.n_samples)  # frequency steps

    i_start = np.where(FREQS >= F_START)[0][0]
    i_end = np.where(FREQS>=F_END)[0][0]

    factor = DecibelToLinear(gainDB)
    fft[i_start:i_end] += factor

#-------------

def AudioSignal_FromFile(path: str, dtype=np.float64) -> AudioSignal:
    out = read(path)
    sample_freq: int = out[0]
    data: np.ndarray = out[1]
  # Parameters
    n_channels = data.ndim  # stereo (2)
    n_samples = data.size
    duration = n_samples / sample_freq
  # Inital data
  # a numpy object from the signal_wave. This will be plotted on the y-axis.
    signal = np.frombuffer(buffer=data, dtype=data.dtype)     # returns all data from ALL channels as a 1-dimensional array. total in array = n_samples * n_channels
  # a numpy object from duration. This will be plotted on the x-axis.
    time = np.linspace(0, duration, num=n_samples)
  # change dtype
    AS = AudioSignal(n_channels, sample_freq, n_samples, time, signal)
    AS.change_dtype(dtype)
  # Return
    return AS