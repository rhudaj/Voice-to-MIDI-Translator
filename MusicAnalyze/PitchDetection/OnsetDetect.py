import numpy as np
from AudioUtil.DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile
from AudioUtil.MusicAnalyze.util import PitchContour, StdDeviation
#----------------------------------------

class OnsetSettings:
    PitchThreshold: float = 2
    MaxPitchThreshold: float = 4
    NumberOfF0sBefore: int = 10
    NumberOfF0sAfter: int = 10
    MinNumberOfFrequenciesBetweenOnsets: int = 10
    timeDistanceBetweenFrames = 0.1

#----------------------------------------

class PitchInfo:
    Onset: bool
    Offset: bool
    Silent: bool
    TransitionNote: bool

#----------------------------------------

SILENT = 0
START_TRANSITION = 1
END_TRANSITION = 2
ONSET = 3
OFFSET = 4


#----------------------------------------

class OnsetDetector:
    #----------------------------------------
    settings: OnsetSettings
    pitches: np.ndarray
    pitchTimes: np.ndarray
    #----------------------------------------

    def __init__(self, AS: AudioSignal):
        y = AS.signal
        fs = AS.sample_freq
        # params
        frame_length = 4096
        window_size = frame_length // 2
        overlap_size = frame_length // 8
        # setup
        self.pitchTimes, self.pitches = PitchContour(y=y, fs=fs, window_size=window_size, overlap_size=overlap_size)
        self.settings = OnsetSettings()

    #----------------------------------------

    def EmptyArr(self, N: int) -> np.ndarray:
        return np.zeros((N,))

    # ----------------------------- HELPER FUNCTIONS (FOR ALGORITHM) -----------------------------

    def GetPitches(self, AS: AudioSignal) -> tuple[np.ndarray]:
        frame_length = 4096
        window_size = frame_length // 2
        overlap_size = frame_length // 8
        #-----------
        x = AS.time
        y = AS.signal
        fs = AS.sample_freq
        #-----------
        self.pitchTimes, self.pitches = PitchContour(y=y, fs=fs, window_size=window_size, overlap_size=overlap_size)
        return [self.pitchTimes, self.pitches]

    def StretchingPitch(self, pitches: np.ndarray , maxPitchThreshold) -> np.ndarray:
        stretch: float = maxPitchThreshold / pitches.max(axis=0)
        return pitches * stretch

    def GetSlopes(self, stretchedPitches: np.ndarray) -> np.ndarray:
        '''
            INPUT:
                - stretchedPitches <ndarray>
                - pitchTimes <ndarray>
                - timeDistanceBetweenFrames <float>
            OUTPUT: pitchSlopes <ndarray>
        '''
        N = stretchedPitches.size
        pitchSlopes = self.EmptyArr(N)
        #----------
        for i in range(1, N-1):
            timeDiff: float = self.pitchTimes[i] - self.pitchTimes[i - 1]
            if timeDiff <= 0: timeDiff = self.settings.timeDistanceBetweenFrames
            diff: float = stretchedPitches[i] - stretchedPitches[i - 1]
            pitchSlopes[i] = diff / timeDiff
        #----------
        return pitchSlopes


    def FollowingSlopes(self, pitchSlopes: np.ndarray) -> np.ndarray:
        '''
            INPUT: pitchSlopes <ndarray>
            OUTPUT: FollowingPitchSlope <ndarray>
        '''
        N = pitchSlopes.size
        FollowingPitchSlope = self.EmptyArr(N)
        #-----
        for i in range(0, N-1):
            sumSlope = pitchSlopes[i]
            #-------
            for j in range(i+1, N-1):
                if ( pitchSlopes[i] * pitchSlopes[j] )  > 0 :   # same sign?
                    sumSlope += pitchSlopes[j]
                else:
                    break
            #-------
            FollowingPitchSlope[i] = sumSlope
        #-----
        return FollowingPitchSlope

    def AvgStdFollowingPitchSlopesInPercent(self, FollowingPitchSlope: np.ndarray, noBefore: int, noAfter: int=0) -> tuple[np.ndarray]:
        N = FollowingPitchSlope.size
        AvgFollowingPitchSlope = self.EmptyArr(N)
        StdFollowingPitchSlope = self.EmptyArr(N)
        #------
        for i in range(noBefore, N-noAfter - 1):
            localPitches: np.ndarray = FollowingPitchSlope[i - noBefore :  i + noAfter + 1]   # *** + 1 ??
            avg: float =  np.mean(localPitches)
            std: float = StdDeviation(localPitches, avg, True)
            AvgFollowingPitchSlope[i] = avg
            StdFollowingPitchSlope[i] = std
        #------
        return [AvgFollowingPitchSlope, StdFollowingPitchSlope]

    # ----------------------------- HELPERS V2 -----------------------------

    def FillOnsetPercentages(self, Status: np.ndarray, StdFollowingPitchSlope: np.ndarray, AvgFollowingPitchSlope: np.ndarray, FollowingPitchSlope: np.ndarray) -> tuple[np.ndarray]:
        N = Status.size
        PercentOnsetBasedPitch = self.EmptyArr(N)
        PercentOffsetBasedPitch = self.EmptyArr(N)
        for i in range(0, N-2):         # N-2 ***
            S1 = (Status[i] == SILENT)
            S2 = (Status[i + 1] == SILENT)
            if not S1 and S2 :
                PercentOnsetBasedPitch[i + 1] = 100
                PercentOffsetBasedPitch[i + 1] = 0
                PercentOnsetBasedPitch[i] = 0
                PercentOffsetBasedPitch[i] = 100
                i+=1
            elif S1 and not S2 :
                PercentOnsetBasedPitch[i] = 0
                PercentOffsetBasedPitch[i] = 100
                PercentOnsetBasedPitch[i + 1] = 100
                PercentOffsetBasedPitch[i + 1] = 0
                i+=1
            elif S1 and S2 :
                PercentOnsetBasedPitch[i + 1] = 0
                PercentOffsetBasedPitch[i + 1] = 0
            else :
                if (    i >= 1 and
                        (   StdFollowingPitchSlope[i - 1] > 0.0000001 or
                            StdFollowingPitchSlope[i - 1] < -0.0000001 ) and
                        (   np.abs(FollowingPitchSlope[i]) >
                            np.abs(AvgFollowingPitchSlope[i - 1] + (StdFollowingPitchSlope[i - 1] * self.settings.PitchThreshold)) )
                ):
                    PercentOnsetBasedPitch[i] = 100
        #--------
        return [PercentOnsetBasedPitch, PercentOffsetBasedPitch]

    def AlteringDetectedOnsets(self, Status: np.ndarray, FollowingPitchSlope: np.ndarray, StdFollowingPitchSlope: np.ndarray, AvgFollowingPitchSlope: np.ndarray):
        N = Status.size
        maxDiff: float
        indexMaxDiff: int
        tempDiff: float
        #----------1
        for i in range(0, N-1):
            if (Status[i] == ONSET and i >= 1):
                maxDiff = np.abs(FollowingPitchSlope[i]) - np.abs((AvgFollowingPitchSlope[i - 1] + (StdFollowingPitchSlope[i - 1] * self.settings.PitchThreshold)))
                indexMaxDiff = i
                #----------
                j = i + 1
                while(j <= i + self.settings.MinNumberOfFrequenciesBetweenOnsets and j < N):
                    if (Status[j] == SILENT): break
                    if (Status[j] == ONSET):
                        tempDiff = np.abs(FollowingPitchSlope[j]) - np.abs((AvgFollowingPitchSlope[j - 1] + (StdFollowingPitchSlope[j - 1] * self.settings.PitchThreshold)))
                        if (tempDiff > maxDiff):
                            # Status[indexMaxDiff].Onset = false;
                            # Status[indexMaxDiff - 1].Offset = false;
                            maxDiff = tempDiff
                            indexMaxDiff = j
                        else:
                            # Status[j].Oset = false;
                            if (Status[j - 1].Offset == True):
                                # Status[j - 1].Offset = False
                                pass
                    j+=1
                #----------
                i = j - 1
                if (i >= N): break
                i += self.settings.MinNumberOfFrequenciesBetweenOnsets - 1

    def AddingTransitions(self, Status: np.ndarray, FollowingPitchSlope: np.ndarray):
        N = Status.size
        for i in range(0, N-1):
            j = i+1
            if(Status[i] == ONSET and Status[i] in (START_TRANSITION, END_TRANSITION)):
                positive1: bool = FollowingPitchSlope[i] > 0
                while j < N:
                    positive2: bool = FollowingPitchSlope[j] > 0
                    if ((positive1 != positive2 or (np.abs(FollowingPitchSlope[j])<0.2)) and j-i>1):
                        Status[j] = ONSET
                        Status[j - 1] = OFFSET
                        # Status[i].TransitionNote = True
                        # pitches[j - 1].TransitionNote = True
                        break
                    j+=1
            #--------
            i = j

    # ----------------------------- MAIN ALGORITHM -----------------------------

    def CalculateOnset(self):
        pitches = self.pitches
        stretchedPitches: np.ndarray = self.StretchingPitch(pitches, self.settings.MaxPitchThreshold)
        pitchSlopes = self.GetSlopes(stretchedPitches)
        FollowingPitchSlope = self.FollowingSlopes(pitchSlopes)
        AvgFollowingPitchSlope, StdFollowingPitchSlope = self.AvgStdFollowingPitchSlopesInPercent(pitches, self.settings.NumberOfF0sBefore, self.settings.NumberOfF0sAfter)
        # ----------
        N = pitches.size

        Status = self.EmptyArr(N)
        PercentOnsetBasedPitch, PercentOffsetBasedPitch = self.FillOnsetPercentages(Status, StdFollowingPitchSlope, AvgFollowingPitchSlope, FollowingPitchSlope)

        for i in range(0, N-1):
            if (PercentOnsetBasedPitch[i] >= 100):
                Status[i] = ONSET
                if (i >= 1): Status[i - 1] = OFFSET
                i += int(self.settings.MinNumberOfFrequenciesBetweenOnsets)

        self.AlteringDetectedOnsets(Status, FollowingPitchSlope, StdFollowingPitchSlope, AvgFollowingPitchSlope)
        self.AddingTransitions(Status, FollowingPitchSlope)


def test():
  AS: AudioSignal = AudioSignal_FromFile('../SampleInput/voiceScale.wav')
  AS.change_dtype(np.float64)
  AS.Normalize()
  OD = OnsetDetector(AS)
  OD.CalculateOnset()

test()