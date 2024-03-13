
def CombineFreqs2Signal(peak_freqs: list[float], duration: float, fs: float):
  print(f'CombineFreqs2Signal, peak_freqs.size = {len(peak_freqs)}')
  signals = []
  for freq in peak_freqs:
    signal = out.create_pure_tone(freq, duration, 1, fs)
    signals.append(signal)

  return out.combine_signals(*signals)

def RecreateNote(signal: AudioSignal) -> AudioSignal:
  '''
    STEPS 1, 2 & 3 COMBINED
    Parameters:
      signal: short section with 1 note played
  '''
  fft = FFT(signal=signal, plotIt=True)
  peak_freqs = fft.PeakFreqIndices(display=True)
  fft.showPlot('Single Note')

  print('FREQ NOTES: ')
  for F in peak_freqs:
    note = closestNoteFromFreq(F)
    print(F, ' ', note)

  return CombineFreqs2Signal(peak_freqs, signal.duration, signal.sample_freq)

def RecreateSample(sig: AudioSignal) -> AudioSignal:
    in_plot = plt.CustomFig('Input Sample')
    in_plot.AddPlot(sig)

  # 2 - GET ONSET TIMES
    onset_indices = get_onset_indices(sig, in_plot)
    in_plot.Show()
    print(f"# onsets = {len(onset_indices)}")

  # 3 - FOR EACH TIME INTERVAL, RECREATE IT AS A DIGITAL SIGNAL. ADD IT TO A TOTAL
    output_signal: np.ndarray = np.zeros(shape=(sig.n_samples))
    output_time: np.ndarray = sig.time

    N = sig.n_samples
    i = 0
    for start_i in onset_indices:
      # RECREATE THE NOTE
        end_i = N-1 if i >= onset_indices.size -1 else onset_indices[i+1]
      # Slice the sample
        slice = copy.deepcopy(sig)
        slice.slice(start_i, end_i)
      # Recreate the sample based on frequencies
        result: AudioSignal = RecreateNote(slice)
      # Add that to the output signal
        output_signal[start_i:end_i] = result.signal
      #increment
        i+=1

  # Create the final result as an AudioSignal
    return AudioSignal(sig.n_channels, sig.sample_freq, sig.n_samples, output_time, output_signal)

def closestNoteFromFreq(freq: float) -> str:
  # find the closest note to this frequency
  min = 1000
  note = 'NA'
  last_diff = min
  #-----------
  for NF in NOTE_FREQS:
    diff = abs(NF[1] - freq)
    if diff < min:
      if diff > last_diff: break  # stop searching
      note = NF[0]
      last_diff = diff
  #-----------
  return note
