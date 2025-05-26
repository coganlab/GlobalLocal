class GroupData:
    data = LabeledArray([])

    def __init__(self, subject_epochs: dict[str, mne.Epochs]):
        new_data = {}
        for sub, epochs in subject_epochs.items():
            new_data[sub] = LabeledArray.from_signal(epochs).to_dict()
        
        combined_sub_channel_data = combine(new_data, (0,2))

        self.data = LabeledArray.from_dict(combined_sub_channel_data)
        self.subjects = set(ch.split("-")[0] for ch in self.data.labels[1])
    
    # @property
    # def subjects(self):
    #     return set(ch.split("-")[0] for ch in self.data.labels[1])
    
seeg = mne.io.read_raw(mne.datasets.misc.data_path() /
                       'seeg' / 'sample_seeg_ieeg.fif')
on1 = trial_ieeg(seeg, 'Fixation', (-1, 1))

group_data = GroupData({'test1': on1})

subjects = group_data.subjects
print(subjects)

group_data.data