# mat
- keys -> header, version, globals, 'DREAMER'

## DREAMER (d)
- d,d[0]  -> len 1
- d[0][0] -> len 10

### d[0][0] (dd)
- dd[0] -> len 1 (Data: {1×23 cell})
- dd[1] -> 128 (EEG_SamplingRate)
- dd[2] -> 256 (ECG_SamplingRate)
- dd[3] -> emotiv sensors
- dd[4] -> 23 (noOfSubjects)
- dd[5] -> 18 (noOfVideoSequences)
- dd[6] -> 'While every care has been taken to...' (disclaimer)
- dd[7] -> 'University of the West of Scotland' (provider)
- dd[8] -> '1.0.2' (version)
- dd[9] -> 'The authors would like to thank...' (Acknowledgement)

#### dd[0] (ddd)
- ddd[0] -> len 23 (EEG data)

##### ddd[0][0] (data)
- data[x] -> persona x

###### data[x][0] (sub)
- subj[0] -> len 7