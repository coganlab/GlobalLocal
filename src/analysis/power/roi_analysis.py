# ongoing refactoring of roi_analysis.ipynb. This will be the main function that imports all of the stats functions and plotting functions.




# i copied this main function from make_epoched_data.py, need to modify this for roi_analysis.py.
def main(subjects=None, task='GlobalLocal', times=(-1, 1.5),
         within_base_times=(-1, 0), base_times_length=0.5, pad_length=0.5, LAB_root=None, channels=None, dec_factor=8, outliers=10, passband=(70,150)):
    """
    Main function to bandpass filter and compute time permutation cluster stats and task-significant electrodes for chosen subjects.
    """
    if subjects is None:
        subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103']

    for sub in subjects:
        bandpass_and_epoch_and_find_task_significant_electrodes(sub=sub, task=task, times=times,
                          within_base_times=within_base_times, base_times_length=base_times_length,
                          pad_length=pad_length, LAB_root=LAB_root, channels=channels,
                          dec_factor=dec_factor, outliers=outliers, passband=passband)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process subjects and plot bandpass-filtered data, compute time permutation cluster matrix of electrodes by time, and find task-significant electrodes.")
    parser.add_argument('--subjects', nargs='+', default=None, help='List of subjects to process. If not provided, all subjects will be processed.')
    parser.add_argument('--task', type=str, default='GlobalLocal', help='Task to process. Default is GlobalLocal.')
    parser.add_argument('--times', type=float, nargs=2, default=(-1, 1.5), help='Time window for event processing. Default is (-1, 1.5).')
    parser.add_argument('--within_base_times', type=float, nargs=2, default=(-1, 0), help='Time window for baseline processing. Default is (-1, 0).')
    parser.add_argument('--base_times_length', type=float, default=0.5, help='Length of the time intervals to randomly select within `within_base_times`. Default is 0.5.')
    parser.add_argument('--pad_length', type=float, default=0.5, help='Length to pad each time interval. Will be removed later. Default is 0.5.')
    parser.add_argument('--LAB_root', type=str, default=None, help='Root directory for the lab. Will be determined based on OS if not provided. Default is None.')
    parser.add_argument('--channels', type=str, default=None, help='Channels to plot and get stats for. Default is all channels.')
    parser.add_argument('--dec_factor', type=int, default=8, help='Decimation factor. Default is 8.')
    parser.add_argument('--outliers', type=int, default=10, help='How many standard deviations above the mean for a trial to be considered an outlier. Default is 10.')
    parser.add_argument('--passband', type=float, nargs=2, default=(70,150), help='Frequency range for the frequency band of interest. Default is (70, 150).')
    args=parser.parse_args()

    print("--------- PARSED ARGUMENTS ---------")
    print(f"args.passband: {args.passband} (type: {type(args.passband)})")
    print(f"args.subjects: {args.subjects} (type: {type(args.subjects)})")

    main(subjects=args.subjects, 
        task=args.task, 
        times=args.times, 
        within_base_times=args.within_base_times, 
        base_times_length=args.base_times_length, 
        pad_length=args.pad_length, 
        LAB_root=args.LAB_root, 
        channels=args.channels, 
        dec_factor=args.dec_factor, 
        outliers=args.outliers, 
        passband=args.passband)