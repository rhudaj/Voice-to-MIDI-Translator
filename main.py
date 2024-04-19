from midiutil import MIDIFile
import numpy as np
from AudioUtil.MidiOutput.MidiOut import signal_to_midi
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.DataStructures.plot import CustomFig
import sys

def run(signal: np.ndarray, srate):
    print("Starting Midi Output...")
    midi:  MIDIFile = signal_to_midi(signal, srate=srate)
    print("Conversion finished!")
    with open ('MidiOutput/MidiOut.mid', 'wb') as file:
        midi.writeFile(file)
    print("Done Midi Output!")


try:
    scriptName, fileName =  sys.argv
except:
    print("Error. Expected 1 argument: <fileName>")
    exit(1)
else:
    AS: AudioSignal = AudioSignal_FromFile(f'SampleInput/{fileName}.wav')
    print("Starting conversion to MIDI...")
    run(signal=AS.signal, srate=AS.sample_freq)
    sys.stdout = sys.__stdout__
    print("DONE. MidiOut.mid file has been created in /MidiOutput")