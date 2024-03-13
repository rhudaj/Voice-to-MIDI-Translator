import AudioUtil.DataStructures.plot as plt
from AudioUtil.DataStructures.AudioSignal import AudioSignal
from pydub import AudioSegment
import numpy as np
from pydub.playback import play
from matplotlib.animation import FuncAnimation
#-------
import threading
import time
from datetime import timedelta
#-------

class AudioAnimation:
  def __init__(self):
    # Audio
    self.audio: AudioSegment = AudioSegment.from_wav('sample.wav')
    audio_fs = self.audio.frame_rate
    self.audio_dur = self.audio.duration_seconds
    # Signal
    self.signal = AudioSignal()
    self.signal.INITfromFile('sample.wav')
    # Plot
    self.fig = plt.CustomFig('Sample.wav')
    self.fig.AddPlot(self.signal)
    # separate thread to play the music
    self.music_thread = threading.Thread(target=play, args=(self.audio,))

  # Function for the animation
  def animate(self, i: int):
    elapsed = (time.perf_counter() - self.music_start)
    i = round(elapsed / self.audio_dur * i)
    self.fig.addLine(i)

  def init(self):
    self.music_thread.start()
    self.music_start = time.perf_counter()

def main():
  audio_anim = AudioAnimation()
  anim = FuncAnimation(
    audio_anim.fig.figure,
    audio_anim.animate,
    init_func= audio_anim.init,
    interval = 20,
  )

  audio_anim.fig.Show()

if __name__ == '__main__':
    main()