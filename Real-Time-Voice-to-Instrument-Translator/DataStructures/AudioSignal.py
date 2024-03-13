import numpy as np
from scipy.io.wavfile import write
import scipy.io.wavfile as wavfile


class AudioSignal:
  n_channels: int
  sample_freq: int # samples/second
  n_samples: int # total samples
  duration: float # in seconds (n_samples/sample_freq)
  # Data:
  time: np.ndarray
  signal: np.ndarray
  dtype = np.float16

  def __init__(self, n_channels=None, sample_freq=None, n_samples=None, time: np.ndarray|None=None, signal: np.ndarray|None=None) -> None:
    # 2 WAYS TO INSTANTIATE THE OBJECT
    if (n_channels==None or sample_freq==None or n_channels==None): return
    # create object from parameters
    self.n_channels = n_channels
    self.sample_freq = sample_freq
    self.n_samples = n_samples
    self.duration = n_samples / sample_freq
    self.time = time
    self.signal = signal
    if type(signal)==None: self.dtype = signal.dtype

  def INITfromFile(self, path: str, printDetails: bool = False):
    out = wavfile.read(path)
    self.sample_freq: int = out[0]
    data: np.ndarray = out[1]
    dtype = data.dtype
    # Parameters
    self.n_channels = data.ndim  # stereo (2)
    self.n_samples = data.size
    self.duration = self.n_samples / self.sample_freq
    # Inital data
    dtype = data.dtype

    if(printDetails):
      print(f'Input file {path}:')
      print(f'\tn_channels = {self.n_channels}')
      print(f'\tduration = {self.duration}')
      print(f'\tn_samples = {self.n_samples}')
      print(f'\tdata.size = {data.size}')
      print(f'\tdtype = {data.dtype}')

    # a numpy object from the signal_wave. This will be plotted on the y-axis.
    self.signal = np.frombuffer(buffer=data, dtype=dtype)     # returns all data from ALL channels as a 1-dimensional array. total in array = n_samples * n_channels
    self.dtype = self.signal.dtype

    # a numpy object from duration. This will be plotted on the x-axis.
    self.time = np.linspace(0, self.duration, num=self.n_samples)

  def output_wav(self, name: str):
    write(filename=f'{name}.wav', rate=self.sample_freq, data=self.signal)

  def Normalize(self):
    # clip signal (y values) to [-1, 1]
    self.signal = np.clip(self.signal, -1, 1)

  def get_derivative(self):
    dx = 1 / self.sample_freq
    derivative = self # copy
    derivative.signal = np.diff(self.signal, prepend=self.signal[0])  #prepend value so that y does not lose size by 1
    return derivative

  def slice(self, i_start: int, i_end: int):
    print(f'slice()')
    start_t = self.time[i_start]
    print(f'\t start_t = {start_t}')
    end_t = self.time[i_end]
    print(f'\t end_t = {end_t}')

    self.time = self.time[i_start:i_end]
    self.signal = self.signal[i_start:i_end]
    self.duration = end_t - start_t
    self.n_samples = i_end - i_start
    self.time -= start_t

  def to_int16(self):
    if(self.dtype == np.int16): return

    peak_value = np.iinfo(np.int16).max
    self.signal = (self.signal * peak_value).astype(np.int16)

    print(f'Changed dtype: ${self.dtype} --> ${np.int16} [max = ${peak_value}]')

  def change_dtype(self, dtype):
    if(self.dtype == dtype): return

    old_peak = np.iinfo(self.dtype).max if np.issubdtype(self.dtype, np.integer) else 1
    new_peak = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    factor = min(old_peak, new_peak) / max(old_peak, new_peak)

    self.signal = (self.signal * factor).astype(dtype) # convert

    print(f'Changed dtype: ${self.dtype} --> ${dtype}')
    self.dtype = dtype
