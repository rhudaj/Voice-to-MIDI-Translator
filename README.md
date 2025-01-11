# Voice-to-Midi Translator

A Python-based project that converts human singing into a MIDI file, enabling direct translation of melodies into digital music. This tool simplifies the process of capturing musical ideas without the need for physical instruments or extensive manual transcription.

## Motivation
Inspired by the challenges of quickly translating vocal melodies into MIDI for music production, this project aims to provide a fast and efficient way to convert vocal input into digital music notation.

## Features
- **Audio Preprocessing**: Enhances vocal prominence and reduces background noise.
- **Note Onset and Offset Detection**: Identifies the start and end of notes in a vocal recording.
- **Pitch Detection with YIN Algorithm**: Determines fundamental frequency for accurate note identification.
- **MIDI Output Generation**: Converts detected notes into a standard MIDI file.

## Dependencies
- `numpy`: Efficient numerical operations
- `librosa`: Audio analysis helper functions
- `scipy`: Input handling for `.wav` files
- `MidiUtil`: MIDI file generation

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script with:

```bash
python main.py --input <path-to-input-file> --output <output-path> --DEBUG
```

Options:

- --input: Path to the input audio file
- --output: Path for the output MIDI *(.mid)* file
- --DEBUG: (optional) enabled debug mode, which logs out more detailed messages to the console.

## How It Works

1. Audio Signal Processing: Converts audio to time and amplitude data.
2. Onset and Pitch Detection: Uses spectral and pitch-based novelty functions.
3. Note Information Extraction: Identifies pitch, duration, and volume for each note.

## Limitations and Future Improvements

### Limitations

1. MIDI Translation: Encodes note data into MIDI format.
Limitations
2. Designed for single-channel audio with monophonic singing.
Performance may vary depending on vocal style and clarity.

###  Future Improvements

1. Real-time processing
2. Dynamic parameter adjustment using machine learning
3. Enhanced user interface


## References

For detailed insights and research references, consult the project report.
