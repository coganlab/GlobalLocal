Global Local Task 

Contact: Jim Zhang and Raphael Geddert

jim.zhang@duke.edu, raphael.geddert@duke.edu

Last edited: 01/16/2024

**Analysis Steps**
### Windows FSL
	• Need to open xquartz on windows before running fsl in the ubuntu app.
		○ https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows
			▪ When running XLaunch, it is critical to deselect Native OpenGL and select Disable access control in the Extra Settings panel
		○ https://superuser.com/questions/1372854/do-i-launch-the-app-xlaunch-for-every-login-to-use-gui-in-ubuntu-wsl-in-windows
			▪ Need to run the line, export DISPLAY=:0 in Ubuntu first before running fsl command for gui to work.
	• Also need to mount the Z: drive on ubuntu every time we open it wtf...
		○ Run this command every time: sudo mount –t drvfs Z: /mnt/Egner
		○ Now to get to this folder, do cd /mnt/Egner. In the FSL gui, it should also be /mnt/Egner
	• To make the inputs to paste, run the makeInputsForFSL.ipynb script that's in the GlobalLocal folder right now, changing the subjects range.
	• Then, open fsl feat in ubuntu and do emacs fslSecondLevelInputs.txt, and highlight all and do edit -> copy. Then can paste this as input into the fsl feat input window. Also change the number of cope images in the GUI.
To make the EV matrix, run the next cell in makeInputsForFSL.ipynb![image](https://github.com/coganlab/GlobalLocal/assets/46616647/9ee8ff82-e97c-4da5-84ed-0037e39fe9ba)

### Preprocessing
1. Run first three cells of plot_clean.ipynb to do line-noise filtering
### Wavelets
1. Run 1st and 3rd cell of copy_wavelet_spec.ipynb to generate .h5 wavelet files, make sure to edit subject and filename in the third cell based on what subject you want and what baseline you're using.
2. Run all cells in plot_wavelets.ipynb to make wavelet plots, saved to filename = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', subj, f'{output_name}-tfr.h5'). Layout is Box/Coganlab
 
### High Gamma Filter and Permutation Testing
1. Run first cell (working version 12/1) to do high gamma filter and permutation testing, with baseline as 1 second before stimulus onset and mirrored to break up fixation cross onset. Using these lines - ```sig1 = HG_ev1._data
sig2 = HG_base._data
sig3 = make_data_same(sig2, (sig2.shape[0],sig2.shape[1],sig2.shape[2]+1)) # originally we want to make the baseline the same shape as the signal. We still want to do that, but first, we'll make it bigger to reflect it once, then back to normal to randomly offset it and remove fixation cross effects.
sig4 = make_data_same(sig3, sig2.shape) #here we do the random offset, we know that sig3 is bigger than sig1 by 1 in the time dimension so it will get randomly sliced.
sig5 = make_data_same(sig4, sig1.shape) #and now sig4 should be sig2 but with a random offset, and we can then set it equal to sig1's shape like the original plan.``` Make sure to edit sub, event, and output_name.
2. Run last few cells to make grid plots for each channel (everything after "ok make greg significance and high gamma combined plots")
**Notes for Experimenter**

**Experiment Procedure**

Run Master\_Script.m to run the experiment. Specify a subject ID and whether to include practice or proceed immediately to the main task. The practice version will ask if you want to proceed to the main task or exit the experiment when it finishes.

Unique data folders are created for each new participant (i.e., never seen before ID, **not** case sensitive), wherein are saved data files for all practice tasks (each iteration) and main task. If the participant has been run before (e.g., just did practice earlier) a new folder will be created for that participant, with the date and time appended to the folder name. 

Press **escape** at any point during a task to **pause the task**. From there, press any button to resume the task, or press **escape** again to **exit the experiment**.

Press **escape** at any point during the instructions to **exit the experiment.**

Press **escape** at any point during the inter block feedback screens to **exit the experiment.**

**This experiment saves almost all progress.** Task data is saved immediately after a trial is completed, and if you exit the experiment in the middle of the main tasks, any completed blocks won’t need to be repeated next time. The exceptions are that the entire practice needs to be completed, so if you exit the task in the middle of the practice, the whole practice will need to be completed again. Likewise, if you exit the experiment in the middle of a block during the main experiment, you will need to repeat that block.

The practice consists of 3 practice sections, each 16 trials long. First, participants practice one task (either global or local) and then practice the other. Last, participants practice combining both the local and global tasks together, responding based on the color of the frame surrounding the letters. Participants must get at least pracAccCutoff  (set to 75) % correct to move on to the next section, otherwise they will need to repeat that practice section. If the practice task is exited prematurely, participants will need to recomplete all three practice sections.

The main experiment consists of 4 blocks, in a counterbalanced block order. Try to avoid pausing for too long in the middle of a block if possible. For example, if a participant is only able to complete 2 of the 4 blocks in a session, the script will automatically resume with the 3<sup>rd</sup> block the next time that participant is run (**assuming the subject ID is exactly the same**). If the experiment is exited in the middle of a block, however, the participant will have to repeat that block.

If the participant was already completed all 4 blocks previously, you will be asked to confirm rerunning all 4 blocks of the task. The old data will still be saved (in whatever folders they were saved in) but the file that stores which runs have been run before will be overwritten to allow new runs to commence.

**Stimulus/Experiment Details**

Stimulus Timing/Trial Count Details can be specified in Master\_Script.m, starting at line 71. Defaults are as follows:

Stimulus Interval (stimInterval) = 1.5 seconds.

- How long stimulus is on the screen and participant is able to respond. Increasing this time (to 2, or 2.5s at most) will (probably) make the task easier, though this hasn’t been confirmed. The task will automatically proceed to the next screen (feedback) as soon as the participant responds.

Fixation Cross Interval (fixInterval) = 0.5 seconds

- If the trial is the first of a section (or the first after pausing the task), the fixation interval is 2 seconds longer than fixIinterval so the participant has time to prepare.

Number of Blocks (numBlocks) = 4

- This should never be changed due to study design.

Trials Per Block in Main Task (trialsPerBlock) = 128

- This can be changed by increments of 16 trials. With a stiminterval of 1.5s, fixInterval of 0.5 seconds, trials run at longest for  3.4 seconds, so 128 trials x 4 blocks x 3.4 seconds = 29 minutes time spent performing main task (not including block breaks or instructions). This can be considerably shorter however, since trials proceed automatically when the participant responds. Assuming an average RT of ~1000 ms, experiment run time (not including breaks and instructions) = ~25 minutes.

Practice Accuracy Cutoff (pracAccCutoff) = 75

- Minimum accuracy percent (out of 100) needed to move on from one practice task to the next. 

