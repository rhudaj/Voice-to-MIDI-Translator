import numpy as np
import librosa
import AudioUtil.DataStructures.plot as plt
import matplotlib.pyplot as pyplot
from matplotlib.axes import Axes
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.DataStructures.note_list import NOTE_FREQS
from AudioUtil.DataStructures.FFT import FFT
import AudioUtil.DataStructures.output as out
import copy

# ------

plot = False

if plot:
  fig, ax = pyplot.subplots(nrows=3, sharex=True)
  ax: tuple[Axes] = ax

# ------

# ------ USING LIBROSA FUNCTIONS

def LogPowerMelSpectrogram(AS: AudioSignal, n_fft, win_length, hop_length, n_mels) -> tuple[np.ndarray, np.ndarray]:
  y = AS.signal
  sr = AS.sample_freq

  # Get Mel Spectogram
  mel_spect = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mels=n_mels,
    # additional args for mel filter bank parameters:
    fmin=100,
    fmax=8000
  )
  log_power_mel_spect = librosa.power_to_db(S=mel_spect, ref=np.max)    # Convert to Log-Power (dB)

  # Array of time values to match the time axis from a feature matrix
  times = librosa.times_like(
    X=mel_spect,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length
  )

  return [log_power_mel_spect, times]

def OnsetEnvelope(AS: AudioSignal) -> tuple[np.ndarray, int]:

  # 1 - Remove frequencies we don't care about. We don't want these contributing to the energy of the STFT
  y = AS.signal
  sr = AS.sample_freq

  # 1 -- Mel Power Spectrogram

  n_fft = 2048                    # in speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz
  win_length = n_fft              # default
  hop_length = win_length // 4    # default
  n_mels = 128

  mel_spect, times = LogPowerMelSpectrogram(AS, n_fft, win_length, hop_length, n_mels)

  # 2 --  spectral flux onset strength envelope

  onset_env = librosa.onset.onset_strength(
    sr=sr,
    S=mel_spect
  )
  onset_env /= onset_env.max()    # Normalize it

  # 3 -- PLOT
  if plot:
    librosa.display.specshow(mel_spect, sr=sr, win_length=win_length, hop_length=hop_length, y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set(title='Log Power Spectrogram')
    ax[0].label_outer()
    ax[1].plot(times, onset_env)
    ax[1].set(title="spectral flux onset strength envelope")
    ax[2].plot(AS.time, AS.signal)

  return [onset_env, hop_length]

# -------- CUSTOM PROCESS - NOVELTY ENVELOPE

def BasicSpectralNovelty(Y) -> np.ndarray:
  # no parameters to play with
  Y_diff = np.diff(Y, n=1)  # Y(n+1,k) - Y(n,k)
  Y_diff[Y_diff < 0] = 0
  nov = np.sum(Y_diff, axis=0)  # sum over all frequencies
  nov = np.concatenate((nov, np.array([0])))  # 1 axis
  return nov

def LocalAverage(Y, M: float) -> np.ndarray:
  '''
  Compute local average of signal
  Args:
      x (np.ndarray): Signal
      M (int): Determines size (2M+1) in samples of centric window  used for local average
  Returns:
      local_average (np.ndarray): Local average signal
  '''
  L = len(Y)
  local_average = np.zeros(L)
  for m in range(L):
    a = max(m - M, 0)
    b = min(m + M + 1, L)
    local_average[m] = (1 / (2 * M + 1)) * np.sum(Y[a:b])
  return local_average

def NoveltySpectrum(AS: AudioSignal, n_fft, win_length, hop_length, fmin = 100, fmax = 8000) -> AudioSignal:

  y = AS.signal
  sr = AS.sample_freq

  # STEP 1 -- STFT

  STFT = librosa.stft(    # complex x frequency x time
    y = y,
    n_fft = n_fft,
    hop_length=hop_length,
    win_length=win_length
  )

  TIMES = librosa.times_like(   # get the time portion of the matrix as an array
    X=STFT,
    sr=sr,
    hop_length=hop_length,
    n_fft=n_fft
  )

  # STEP 2 -- FREQUENCY RESTRICT
  N_BINS = int(1 + (n_fft / 2))
  print(f'N_BINS = {N_BINS}')
  STFT_FREQS = np.arange(0, N_BINS) * sr / n_fft      # bin number -> freq
  i0 = np.where(STFT_FREQS >= fmin)[0][0] - 1
  i1 = np.where(STFT_FREQS >= fmax)[0][0]
  STFT[0:i0] = 0
  STFT[i1:N_BINS-1] = 0

  # STEP 1 -- Spectral Novelty Function + Compression

  MAGS = np.abs(STFT)
  gamma = 100   # how much to compress?
  Y = np.log(1 + gamma * np.abs(MAGS))
  nov = BasicSpectralNovelty(Y)    # spectral-based novelty function:


  # STEP 3 - SUBTRACTING LOCAL AVERAGE

  M_sec = 0.1
  Fs_nov = sr / hop_length
  M_samps = int(np.ceil(M_sec * Fs_nov))
  nov = nov - LocalAverage(Y=nov, M=M_samps)
  nov[nov < 0] = 0

  # STEP 4 - NORMALIZE

  nov /= nov.max()

  # PLOT

  if plot:
    librosa.display.specshow(Y, sr=sr, win_length=win_length, hop_length=hop_length, y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set(title='Log Power Spectrogram, Compressed')
    ax[0].label_outer()

    ax[1].plot(TIMES, nov)
    ax[1].set(title="Spectral Based Novelty 1")

  return AudioSignal(AS.n_channels, Fs_nov, n_samples=TIMES.size, time=TIMES, signal=nov)

def NoveltyEnvelope(AS: AudioSignal) -> tuple[np.ndarray, int]:
  n_fft = 4096
  win_length = n_fft
  hop_length = win_length // 1
  # ------------------------
  nov = NoveltySpectrum(AS, n_fft, win_length, hop_length)
  nov_env = nov.SignalEnvelope()
  # ------------------------
  return [nov_env, hop_length]

# ------ ONSETS

def get_onset_indices(AS: AudioSignal):
  '''
    A sample n is selected as an peak if the corresponding x[n] fulfills the following three conditions:

      x[n] == max(x[n - pre_max:n + post_max])

      x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta

      n - previous_n > wait
  '''
  nov_env, hop_length = NoveltyEnvelope(AS)

  onset_samples = librosa.onset.onset_detect(
    onset_envelope = nov_env,
    units='samples',              # return indicies
    backtrack=True,               # If True, detected onset events are backtracked to the nearest preceding minimum of energy.
    normalize=True,               # (default) already normalized
    hop_length=hop_length,
    # optional params for peak picking algorithm
    pre_max=1,
    post_max=1,
    pre_avg=4,
    post_avg=4,
    wait=5
  )

  return onset_samples


def test1():
  AS: AudioSignal = AudioSignal_FromFile('../SampleInput/voiceScale.wav')
  AS.change_dtype(np.float64)

  onset_indices = get_onset_indices(AS)


  if plot:
    ax[2].plot(AS.time, AS.signal)
    ax[2].set(title="Input Signal")
    for sample_num in onset_indices:
      ax[2].axvline(AS.time[sample_num], color='red')
