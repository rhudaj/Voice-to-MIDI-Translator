#python imports
import argparse # to parse cmd-line args
import os, sys
#external imports
from midiutil import MIDIFile
#internal imports
from MidiOutput.MidiOut import signal_to_midi
from DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile

def main():

    # 1) parse command line arguments:

    parser = argparse.ArgumentParser(description="Convert voice audio input to MIDI format.")

    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file (.wav)")
    parser.add_argument("--output", type=str, required=True, help="Path for the output MIDI file (.mid)")
    parser.add_argument("--DEBUG", action="store_true", help="Enable debug mode for detailed output")

    args = parser.parse_args()

    # 2) Validate arguments

    # validate input file existence
    if not os.path.isfile(args.input):
        parser.error(f"The input file '{args.input}' does not exist.")

    # 3) do the transformation

    print("Starting conversion to MIDI...")

    midi:  MIDIFile = signal_to_midi(args.input, args.DEBUG)

    print("Conversion finished!")
    print("Starting Midi Output...")

    with open (args.output, 'wb') as file:
        midi.writeFile(file)

    print(f"DONE. .mid file has been created at: {args.output}")

if __name__ == "__main__":
    main()