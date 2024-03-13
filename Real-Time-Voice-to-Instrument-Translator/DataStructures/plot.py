import matplotlib.pyplot as plt
from AudioUtil.DataStructures.AudioSignal import AudioSignal
import numpy as np

class CustomFig:
  figure: plt.Figure
  w, h = 10, 8

  def __init__(self, title: str) -> None:
    self.figure = plt.figure(figsize=(self.w, self.h))
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('time (s)')

  def AddPlot(self, signal: AudioSignal):
    plt.plot(signal.time, signal.signal)
    plt.xlim(0, signal.duration)

  def PlotFFT(self, fft: np.ndarray, F: np.ndarray, maxFreq=2000):
    # fft : array of points (real and imaginary parts)
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency')
    plt.xlim(0, maxFreq)
    plt.plot(F, fft)

  def AddSpectogram(self, signal: AudioSignal):
    # visual representation of the signal strength at different frequencies
    # showing us which frequencies dominate the recording as a function of time:
    # vmin and vmax are chosen to bring out the lower frequencies that dominate this recording
    plt.specgram(x=signal.signal, Fs=signal.sample_freq, cmap = plt.cm.get_cmap('bone'))
    plt.title('Spectogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, signal.duration)
    plt.colorbar()
    plt.show()

  def AddSpectogram2(self, signal: AudioSignal):
    fft = np.fft.fft(signal.signal)
    freq = np.fft.fftfreq(fft.shape[-1])

    ax = plt.subplot()
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.plot(freq, fft.real, freq, fft.imag)


  def addPoint(self, x: float, y: float):
    plt.plot(x, y, 'ro')  # 'o' can be used to only draw a marker. 'r' = red

  def addLine(self, x: float):
    plt.axvline(x, color="red")

  def Show(self):
    plt.show()
