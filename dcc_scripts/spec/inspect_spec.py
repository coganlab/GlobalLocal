import mne
import numpy as np
import os

#file_path = '/Users/erinburns/Library/CloudStorage/Box-Box/CoganLab/BIDS-1.1_GlobalLocal/BIDS/derivatives/spec/multitaper/D0059/CorrectTrials_Stimulus_Locked_rescaled-tfr.h5'
file_path = '/cwork/etb28/BIDS-1.1_GlobalLocal/BIDS/derivatives/spec/multitaper/D0059/CorrectTrials_Stimulus_Locked_rescaled-tfr.h5'
tfr = mne.time_frequency.read_tfrs(file_path)
freq_values = tfr.freqs
print(freq_values)
#print(tfr.data.shape)