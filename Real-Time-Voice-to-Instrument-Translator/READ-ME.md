# Project Title

This project is a library of custom functions and data structures to work with audio files, create signals and process them in a variety of ways. It serves as a basis for creating cool projects that involes audio signals. 

## Description

The main data structure used in this library is ***AudioSignal***. This is a reprentation of sound in a digital format. It is a wrapper for a 2D numpy array (`ndarray`) with additional information such as: sample frequency, time duration, number of channels, etc. 

Using this digital representation of an audio signal, the library has a variety of functions for processing the data. These features include, but not limited to: generating FFT's, offset/transient detection, peak detection, generating pure-tones, triangle and square waves. 

The library also provides a wrapper for 'matplotlib.pyplot' objects, allowing you to display signals in a variety of ways. 

## Getting Started

### Dependencies

***Prerequisites***
* `Python` installed
*   `ffmpeg` installed (to play audio clips using `pydub` library)

***Libraries***
* `numpy`
* `matplotlib`
* `scipy`
*  `librosa`

### Installing

To download this library: 
```
git pull https://github.com/rhudaj/Python-Audio-Processing-Library.git
```

To download the dependet libraries (listed above): 
```
pip install <package-name>
````

### Executing program

Refer to the instructions commented in the source code 

## Help

For help, feel free to contact me. Click on my profile for my contact information.

## Authors

Roman Hudaj

## Version History

N/A

## License

N/A

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
