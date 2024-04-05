from midiutil import MIDIFile
import numpy as np
from AudioUtil.MidiOutput.MidiOut import signal_to_midi
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile

def run(signal: np.ndarray, srate):
    print("Starting Midi Output...")
    midi:  MIDIFile = signal_to_midi(signal, srate=srate)
    print("Conversion finished!")
    with open ('MidiOut.mid', 'wb') as file:
        midi.writeFile(file)
    print("Done Midi Output!")

name = 'voiceScale'
AS: AudioSignal = AudioSignal_FromFile(f'../SampleInput/{name}.wav')
y: np.ndarray = AS.signal
run(signal=y, srate=AS.sample_freq)