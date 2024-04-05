import sys
from enum import Enum
import os
import numpy as np
import librosa
import midiutil
import matplotlib.pyplot as plt
from matplotlib import colors

DEBUG = True
PLOT = True

if DEBUG:   sys.stdout = sys.__stdout__
else:       sys.stdout = open(os.devnull, 'w')

class Settings:
    # USER SET CONSTANTS
        note_min: str = "C1"            # Lowest allowed note in "A#4" format. Defaults to "A2".
        note_max: str = "E5"            # Highest allowed note in "A#4" format. Defaults to "E5".
        frame_length: int = 4096        # Frame length for analysis. Defaults to 2048.
        hop_length: int = 1024          # Hop between two frames in analysis. Defaults to 512.
        # PYIN Probabilities
        pitch_acc: float = 0.9          # Probability (reliability) that the pitch estimator is correct.
        voiced_acc: float = 0.9         # Estimated accuracy of the "voiced" parameter.
        onset_acc: float = 0.9          # Estimated accuracy of the onset detector.
        spread: float = 0.2             # Probability that the audio signal deviates by one semitone due to vibrato or glissando.
        # Transitional Probabilities
        p_stay_note: float = 0.9        # Probability of staying in the SAME note for two subsequent frames. Defaults to 0.9.
        p_stay_silence: float = 0.7     # Probability of staying in the silence state for two subsequent frames. Defaults to 0.7.
    # DERIVED FROM ABOVE
        # min/max note conversion
        fmin = librosa.note_to_hz(note_min)
        fmax = librosa.note_to_hz(note_max)
        midi_min = librosa.note_to_midi(note_min)
        midi_max = librosa.note_to_midi(note_max)
        # totals
        n_notes = midi_max - midi_min + 1
        n_states = n_notes * 2 + 1

# ------------ HELPERS ------------

class NoteInfo:
    '''
        onset_time :    float
        offset_time :   float
        pitch :         float
        note_name :     str
        volume :        int
    '''
    def __init__(self, onset_time: float, offset_time: float, pitch, note_name: str, volume: int = 100) -> None:
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.pitch = pitch
        self.note_name = note_name
        self.volume = volume

    def as_str(self, i):
        return f'{i}, {self.offset_time}, {self.onset_time}, {self.note_name}, {self.pitch}, {self.volume}'

class MidiNote:
    '''
        start:      float
        duration:   float
        volume:     int
        pitch :     float
    '''
    def __init__(self, piano_note: NoteInfo, bpm: float, Qfraction: float):
        note_dur = GetNoteDur(bpm, Qfraction)
        self.start = piano_note.onset_time / note_dur
        offset_time = piano_note.offset_time / note_dur
        self.duration = offset_time - self.start
        self.pitch =  int(piano_note.pitch)
        self.volume = piano_note.volume

class Observation(Enum):
    SILENCE = 0
    ONSET = 1
    SUSTAIN = 2

def GetNoteDur(bpm: float, fraction = 1/4):
    '''
        Half note               =  120 / BPM
        Quarter note            =   60 / BPM
        Eighth note             =   30 / BPM
        Sixteenth note          =   15 / BPM
    '''
    return (240 * fraction) / bpm

def Pnot(p): return 1 - p

def note2state(note): return (2 * note) + 1

def state2note(state): return (state - 1) / 2

#---------- OUTPUT CHECKING

def write_pianoroll(pianoroll: list[NoteInfo]):
    with open('pianoroll_correct.txt', 'w') as f:
        for i, note in enumerate(pianoroll):
            f.write(note.as_str(i) + '\n')

def compare(pianoroll: list[NoteInfo]):
    print('comparing correct/new pianoroll:')
    with open('pianoroll_correct.txt', 'r') as f:
        lines: list[str] = f.readlines()
        if len(lines) != len(pianoroll):
            print(f'\t Diff # notes: correct: {len(lines)}, new: {len(pianoroll)}')
        for i, line in enumerate(lines):
            note = pianoroll[i]
            note_as_str = note.as_str(i)
            if note_as_str != line.rstrip():
                print(f'\t Diff notes:')
                print(f'\t\t Correct: {line}')
                print(f'\t\t New    : {note_as_str}')
    print('\tDone')

#----------------------

def transition_matrix() -> np.ndarray:
    """
    Returns the transition matrix with 1 silence state and 2 states (onset and sustain) for each note.
    This matrix mixes an acoustic model with two states with an uniform-transition linguistic model.

    Returns
    -------
    TRANSITIONS : np.ndarray
        - shape: ( 2 * n_notes + 1 , 2 * n_notes + 1 )
        - transmat[i,j] = P(state i -> state j)
    """
    S = Settings()

    p_l = Pnot(S.p_stay_silence) / S.n_notes        # p_l <- P(not staying in silence, from state i to j)
    p_ll = Pnot(S.p_stay_note) / (S.n_notes + 1)    # p_ll <- P(not staying on the same note, from state i to j)

    # Transition matrix:
    TRANSITIONS = np.zeros((S.n_states, S.n_states))

    # State 0: silence
    TRANSITIONS[0, 0] = S.p_stay_silence
    for i in range(S.n_notes):
            S1 = note2state(i)
            S2 = S1 + 1
            TRANSITIONS[0, S1] = p_l
        # States 1, 3, 5... = onsets
            TRANSITIONS[S1, S2] = 1
        # States 2, 4, 6... = sustains
            TRANSITIONS[S2, 0] = p_ll
            TRANSITIONS[S2, S2] = S.p_stay_note
            for j in range(S.n_notes):
                S3 = note2state(j)
                TRANSITIONS[S2, S3] = p_ll

    if DEBUG:
        print('transition_matrix:')
        print(f'\t n_notes = {S.n_notes}')
        print(f'\t P(not staying in silence, from state i to j) = {p_l}')
        print(f'\t P(not staying on the same note, from state i to j) = {p_ll}')

    if PLOT:
        cmap = colors.ListedColormap(['red', 'blue', "green", "orange"])
        bounds= [S.p_stay_silence, S.p_stay_note, p_l, p_ll]
        bounds.sort()
        bounds.append(1)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(TRANSITIONS, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)   # tell imshow about color map so that only set colors are used
        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)         # make a color bar
        plt.show()

    return TRANSITIONS

def prior_probabilities(audio_signal: np.ndarray, srate: int) -> np.ndarray:
    """
    Estimate prior (observed) probabilities from audio signal

    Parameters
    ----------
    audio_signal : np.ndarray

    Returns
    -------
    priors : 2D numpy array.

        - shape: (n_states, # frames)
        - Sequence of likelihoods, prob[s, t] = likelihood
        - of seeing the observation at time t from state s.
    """
    S = Settings()

    print('prior_probabilities:')
    print('\t STARTING pyin(...)')

    # pitch and voicing
    PITCH, VOICED_FLAG, voiced_prob = librosa.pyin(
        y=              audio_signal,
        fmin=           S.fmin * 0.9,
        fmax=           S.fmax * 1.1,
        sr=             srate,
        frame_length=   S.frame_length,
        win_length=     S.frame_length // 2,
        hop_length=     S.hop_length
    )

    print('\t END pyin(...)')

    tuning = librosa.pitch_tuning(PITCH)    # float in [-0.5, 0.5)

    print(f'\t tuning = {tuning}')

    PITCH = PITCH - tuning

    PITCH_MIDI = np.round(librosa.hz_to_midi(PITCH)).astype(int)

    ONSETS = librosa.onset.onset_detect(
        y=audio_signal,
        sr=srate,
        hop_length=S.hop_length,
        backtrack=True
    )

    PRIORS = np.ones((S.n_states, len(PITCH_MIDI)))

    for i, flag in enumerate(VOICED_FLAG):
        PRIORS[0, i] =  Pnot(S.voiced_acc) if flag else S.voiced_acc

    for i, pitch in enumerate(PITCH_MIDI):
        isOnset: bool = (i in ONSETS)
        for note in range(S.n_notes):
                s1 = note2state(note)
                s2 = s1 + 1
            # State 1
                PRIORS[s1, i] = S.onset_acc if isOnset else Pnot(S.onset_acc)
            # State 2
                pitch2 = note + S.midi_min
                diff = np.abs(pitch2 - pitch)
                if      pitch2 == pitch:    result = S.pitch_acc
                elif    diff == 1:          result = S.pitch_acc * S.spread
                else:                       result = Pnot(S.pitch_acc)
                PRIORS[s2, i] = result

    if PLOT and False:
        ax = plt.subplot()
        ax: plt.Axes = ax
        T = librosa.times_like(PITCH)
        ax.plot(T, voiced_flag, color='blue')
        ax.plot(T, voiced_prob, color='red')
        ax.plot(T, PITCH_MIDI, color='purple')
        plt.show()

    if PLOT:
        ax = plt.subplot()
        ax: plt.Axes = ax
        img = ax.imshow(PRIORS, interpolation='nearest', origin='lower', cmap='Spectral', aspect='auto')   # tell imshow about color map so that only set colors are used
        plt.colorbar(img)
        plt.show()

    return PRIORS

#----------------------

def states_2_pianoroll(states: list, hop_time: float) -> list:
    """
    Converts state sequence to an intermediate, internal piano-roll notation

    Parameters
    ----------
    states : list of int (or other iterable)
        Sequence of states estimated by Viterbi
    hop_time : float
        Time interval between two states.

    Returns
    -------
    output : List of lists
        output[i] is the i-th note in the sequence.
        Each note is a list described by:
            [onset_time, offset_time, pitch, note_name]
    """
    S = Settings()

    states_ = np.hstack((states, np.zeros(1)))
    pianoroll: list[NoteInfo] = []

    # ITERATE THROUGH STATES
    cur_state = Observation.SILENCE
    last_onset = 0
    last_offset = 0
    last_midi = 0

    print(states_)

    for i, state in enumerate(states_):

        note = state2note(state)

        if cur_state == Observation.SILENCE:

            if int(state % 2) != 0:
                # Found an onset!
                last_onset = i * hop_time
                last_midi = note + S.midi_min
                last_note = librosa.midi_to_note(last_midi)
                cur_state = Observation.ONSET

        elif cur_state == Observation.ONSET:
            if int(state % 2) == 0:
                # found note start!
                cur_state = Observation.SUSTAIN

        elif cur_state == Observation.SUSTAIN:

            if int(state % 2) != 0:
                # Found an onset. Finish last note
                last_offset = i * hop_time
                note_info = NoteInfo(last_onset, last_offset, last_midi, last_note)
                pianoroll.append(note_info)

                # Start new note
                last_onset = i * hop_time
                last_midi = note + S.midi_min
                last_note = librosa.midi_to_note(last_midi)
                cur_state = Observation.ONSET

            elif state == 0:
                # Found silence. Finish last note.
                last_offset = i * hop_time
                note_info = NoteInfo(last_onset, last_offset, last_midi, last_note)
                pianoroll.append(note_info)
                cur_state = Observation.SILENCE

    return pianoroll

def pianoroll_2_MidiFile(pianoroll: list[NoteInfo], bpm: float, Qfraction=1/4) -> midiutil.MIDIFile():
    """ Converts an internal piano roll notation to a MIDI file

        Parameters
        ----------
            bpm: float
                Beats per minute for the MIDI file. If necessary, use
                bpm = librosa.beat.tempo(y)[0] to estimate bpm.

            pianoroll : list
                A pianoroll list as estimated by states_to_pianoroll().

        Returns
        -------
            None.
    """

    midi = midiutil.MIDIFile(1) # initialize midi file obj

    midi.addTempo(track=0, time=0, tempo=bpm)

    for note in pianoroll:
        mn = MidiNote(note, bpm, Qfraction)
        midi.addNote(
            track=0,
            channel=0,
            pitch=mn.pitch,
            time = mn.start,
            duration = mn.duration,
            volume=mn.volume             # can modify this based on the amplitude of the signal
        )

    return midi

def signal_to_midi(audio_signal: np.ndarray, srate: int) -> midiutil.MIDIFile():
    """Converts an audio signal to a MIDI file
    Args:
        audio_signal (np.array):            Array containing audio samples
    Returns:
        midi (midiutil.MIDIFile): A MIDI file that can be written to disk.
    """

    S = Settings()

    # STEP 1
    TRANSITIONS = transition_matrix()

    # STEP 2
    prior_probs = prior_probabilities(audio_signal, srate)  # prob[s,t] (of being note <s> at time t)

    print(f'prior_probs.shape: {prior_probs.shape}')

    # STEP 3
    p_init = np.zeros(S.n_states)     # S.n_states
    p_init[0] = 1
    states = librosa.sequence.viterbi(prior_probs, TRANSITIONS, p_init=p_init)

    print(f'states.shape: {states.shape}, S.n_states = {S.n_states}')

    # STEP 4
    pianoroll: list[NoteInfo] = states_2_pianoroll(states, hop_time = S.hop_length / srate)

    compare(pianoroll)

    # STEP 5
    bpm = int(librosa.beat.tempo(y=audio_signal)[0])

    print(f'\t bpm = {bpm}')

    midi = pianoroll_2_MidiFile(pianoroll, bpm, 1/4)

    # RETURN
    return midi