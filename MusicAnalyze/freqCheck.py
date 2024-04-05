import numpy as np
import os
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.DataStructures.plot import CustomFig
from AudioUtil.MusicAnalyze.util import StdDeviation, RemoveDC, EmptyArr
from scipy.signal import savgol_filter # filter out the noise
from AudioUtil.MusicAnalyze.PitchDetection.YINalg import yin
from AudioUtil.DataStructures.FFT import FFT
import librosa


# ------- TESTS

def testNotes(name: str):

    FIG = CustomFig()
    axes = FIG.plot_bottom(x=[0], y=[0], label_x='Freq(Hz)', label_y='Magnitude', x_min=0, x_max=1000)

    for file in os.listdir('../SampleInput/sungNotes'):
        filename = os.fsdecode(file)
        if not filename.endswith('.wav'): continue
        filename = filename.split('.wav')[0]

        AS: AudioSignal = AudioSignal_FromFile(f'../SampleInput/sungNotes/{filename}.wav')
        AS.change_dtype(np.float64)
        AS.Normalize()

        fft = FFT(AS)

        axes.plot(fft.F, fft.MAGS, label=filename)

    axes.legend()
    FIG.show()

# testNotes('sungNotes/la')


AS: AudioSignal = AudioSignal_FromFile(f'../SampleInput/voiceScale_Filtered.wav')
y = AS.signal
t = AS.signal

n_fft=4096
win_length = n_fft
hop_length = win_length // 4
STFT = librosa.stft(y=y, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # shape=(..., 1 + n_fft/2, n_frames)
STFT = np.abs(STFT)

n_bins = STFT.shape[0]
n_frames = STFT.shape[1]

print(f'n_bins = {n_bins}')
print(f'n_frames = {n_frames}')

FREQS = librosa.fft_frequencies(sr=AS.sample_freq, n_fft=n_fft)

frames = np.arange(0, n_frames-1, 1)
frame_times = [ frame * hop_length / AS.sample_freq for frame in frames ]

F0 = np.zeros(n_frames-1)

for frame in range(n_frames):
    for bin in range(n_bins):
        b = n_bins-bin-1
        if STFT[b][frame] >= 0.6:
            F0[frame-1] = FREQS[b]
            continue


FIG = CustomFig()
ax = FIG.plot_bottom([0], [0], x_min=0, x_max=45, y_min=0, y_max=1000)
img = librosa.display.specshow(
    librosa.amplitude_to_db(STFT, ref=np.max),
    y_axis='log',
    x_axis='time',
    ax=ax
)

ax.plot(frame_times, F0, color='green')

ax.set_title('Power spectrogram')
FIG.show()
