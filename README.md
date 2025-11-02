Global Local Task 

Contact: Jim Zhang and Raphael Geddert

jim.zhang@duke.edu, raphael.geddert@duke.edu

Last edited: 01/16/2024

### BIDS Coding (makes BIDS files)
1. Run makeTrials_GL.m (/Users/jinjiang-macair/Library/CloudStorage/Box-Box/CoganLab/D_Data/GlobalLocal/makeTrials_GL.m) with the subject id (D##) and date (YYMMDD) to create a Trials.mat file for that subject. Need to add makeTrials_GL.m to path as well as MATLAB-env folder (/Users/jinjiang-macair/Documents/MATLAB/MATLAB-env). If MATLAB-env isn't there, you can clone it from https://github.com/coganlab/MATLAB-env
2. Run BIDS_convert_wsl.sh (within BIDS_coding repository, global local branch). Steps 3-5 go into detail on how to do this.
3. To install dependencies, need to ```conda create env environment.yml``` on Mac if not already created, and give it an environment name. Or do ```conda env create -f environment.yml``` from the envs folder if on Windows.
4. Need to ```conda activate BIDS_coding``` or whatever you named the conda environment. 
5. Now cd into the BIDS_converter subfolder within BIDS_coding repository (open WSL, cd /git/BIDS_coding/BIDS_converter), and do ```./BIDS_convert_wsl.sh``` after modifying BIDS_convert_wsl.sh with your chosen SUB_IDS (line 18). Or, BIDS_convert_mac or whichever script fits your OS. NOTE: To open the WSL script, do ```explorer.exe .``` to open the file explorer in that location.
6. Copy the BIDS folder into Box (run it locally because it's faster)
   
### Duke Health VPN
1. Get a Duke Health Enterprise Account (send e-mail to dibs-it@duke.edu asking for them to set this up)
2. Set up multi-factor authentication: https://idms-mfa.oit.duke.edu/mfa/help
3. Follow the instructions to set up a Duke VPN: https://oit.duke.edu/service/vpn/
4. You can test your VPN access and download FortiNAC and necessary antivirus: portal.duke.edu, https://duke.service-now.com/kb_view.do?sysparm_article=KB0034098
5. Open Cisco AnyConnect, and connect to the Duke Health VPN using this address: vpn.duhs.duke.edu
6. Enter your netid and netid password, and type 'push' as the Second Password to send a push notification to your MFA app for authentication.
7. Wait a minute or so for the VPN connection to let you through the firewall.

### Microsoft Remote Desktop
1. Download Microsoft Remote Desktop on your local machine
2. On the remote machine that you want to connect to, get your full PC device name by going to Settings -> System -> About -> Full device name (i.e., NEU-7BTXKH2.dhe.duke.edu)
3. On your local machine, follow these instructions to set up the Duke OIT RDS Gateway on Microsoft Remote Desktop: https://oit.duke.edu/help/articles/kb0032645/. NOTE: On Mac, you need to **check** the 'Bypass for local addresses' option, not uncheck it as the instructions say.
4. On your local machine, open Microsoft Remote Desktop, click Add PC, and put in your full device name as the PC name. Put in your netid and netid password as your User account.
5. On your local machine, connect to Duke Health VPN. Wait a bit for it to let you through the firewall.
6. Now try connecting to your remote machine through Microsoft Remote Desktop by double-clicking the icon for it.
   
### Windows FSL
1. Need to open xquartz on windows before running fsl in the ubuntu app. https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows
2. When running XLaunch, it is critical to deselect Native OpenGL and select Disable access control in the Extra Settings panel. https://superuser.com/questions/1372854/do-i-launch-the-app-xlaunch-for-every-login-to-use-gui-in-ubuntu-wsl-in-windows
3. Need to run the line, export DISPLAY=:0 in Ubuntu first before running fsl command for gui to work.
4. Also need to mount the Z: drive on ubuntu every time we open it. Run this command every time: ```sudo mount –t drvfs Z: /mnt/Egner```
5. Now to get to this folder, do ```cd /mnt/Egner```. In the FSL gui, it should also be ```/mnt/Egner```
6. To make the inputs to paste, run the makeInputsForFSL.ipynb script that's in the GlobalLocal folder right now, changing the subjects range.
7. Then, open fsl feat in ubuntu and do emacs fslSecondLevelInputs.txt, and highlight all and do edit -> copy. Then can paste this as input into the fsl feat input window. Also change the number of cope images in the GUI.
8. To make the EV matrix, run the next cell in makeInputsForFSL.ipynb

### Duke Compute Cluster
1. Download the Remote - SSH Extension on VS Code: https://marketplace.visualstudio.com/items/?itemName=ms-vscode-remote.remote-ssh. Then, set up a remote host from VS Code to dcc-login.oit.duke.edu. Click the >< button on the bottom left and then choose "connect to host", entering dcc-login.oit.duke.edu. It'll ask for a password (enter your net id password) and then 2-step authentication. NOTE: To avoid issues with having to log in multiple times when connecting to remote host from VS code, do ```ssh -Y netid@dcc-login.oit.duke.edu``` from a terminal and then after logging in, open your shell's startup file by doing ```nano ~/.bashrc```, then add ```[[ $- != *i* ]] && return``` as the first line of your shell's startup file, and save/close it by pressing Ctrl+X.
2. ALTERNATIVELY! Set up an ssh key so that you don't have to manually log in: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/login/#ssh-keys - note that you need to ssh into dcc from terminal first before connecting to host on vscode
   
3. You can do ```ssh -Y netid@dcc-login.oit.duke.edu``` from a terminal to access the DCC.
   
4. To be able to access data from scripts on the DCC using ```LAB_root = os.path.join("cwork", "your_net_id")``` -> Move data from /hpc/home/your_net_id/coganlab/Data/BIDS-1.1_GlobalLocal/ to /cwork/your_net_id/BIDS-1.1_GlobalLocal/ using the Duke Compute Cluster (DCC) Data Transfer Node as the collection on Globus (https://app.globus.org/file-manager?destination_id=1ad66c7c-4f60-11e8-900c-0a6d4e044368&destination_path=%2Fcwork%2Fjz421%2FBIDS-1.1_GlobalLocal%2F&origin_id=1ad66c7c-4f60-11e8-900c-0a6d4e044368&origin_path=%2Fhpc%2Fhome%2Fjz421%2Fcoganlab%2FData%2FBIDS-1.1_GlobalLocal%2F&two_pane=true).<img width="1273" alt="Screenshot 2025-06-06 at 1 07 21 PM" src="https://github.com/user-attachments/assets/dd0204bd-3536-4fb2-9333-ba289f02ec4a" />

5. Use git repositories on the DCC to sync code with local computer code. Use the dcc_scripts folder for scripts that will live on the DCC. Make sure to set up an SSH key for permissions, and pull before making any changes (refer to step 4 here: https://github.com/dward2/BME547/blob/main/Assignments/01_tool_setup_git_intro.md). First, git clone this repository to your netid folder under coganlab on the DCC. Then, do git config --global user.email "your_email@email.com" and do git config --global user.name "your_github_username". 

6. To move files from DCC to a local machine or Box using Terminal, on windows, can run something like: ```scp jz421@dcc-login.oit.duke.edu:/cwork/jz421/BIDS-1.1_GlobalLocal/BIDS/derivatives/spec/multitaper/subjects_tfr_objects/*.png C:Users/jz421/Desktop/tfr_figures/``` but replace the paths with where you've saved the figures on the dcc and where you want to save them to. For mac, do ```scp "jz421@dcc-login.oit.duke.edu:/cwork/jz421/BIDS-1.1_GlobalLocal/BIDS/derivatives/spec/multitaper/subjects_tfr_objects/*.png" ~/Desktop/tfr_figures/```

  
# **Analysis Steps**

### Preprocessing
1.	<img width="1360" height="614" alt="image" src="https://github.com/user-attachments/assets/70a10f46-2a2c-4111-8991-943cb3e05f25" />
2. Run first three cells of plot_clean.ipynb to do line-noise filtering (for new subjects, will need to run this twice and exclude the eeg channels from the RuntimeWarning). Or just run src/analysis/preproc/plot_clean.py and pass in the subjects. (i.e., python plot_clean.py --subjects D0057 D0059)
3. Copy Trials.csv from Box/CoganLab/D_Data/GlobalLocal/D### for newly run subjects into Box/CoganLab/D_Data/GlobalLocal/rawDataCopies. Rename as D###_behavioralData.csv.
4. Run makeRawBehavioralData.ipynb to generate accuracy arrays for newly run subjects
### Wavelets
1. Run make_wavelets.ipynb to make wavelet tfr files (mne.TimeFrequency.EpochsTFR), saved to filename = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', subj, f'{output_name}-tfr.h5')
2. Run plot_wavelets.ipynb to make wavelet plots for each electrode
3. Run wavelet_differences.ipynb to make wavelet_difference plots for different conditions.
4. All wavelet functions live in wavelet_functions.py (also, copy_wavelet_spec.ipynb is deprecated, that was a copy of Aaron's old code)
5. Alternatively, use the submit/sbatch script on the cluster.
    
### High Gamma Filter and Permutation Testing
1. Run make_epoched_data.py to do the stats without plotting. Run make_epoched_data.py like this: ```(ieeg) PS C:\Users\jz421\Desktop\GlobalLocal> python make_epoched_data.py --passband 4 8 --subjects D0057```. So the passband needs to pass in the lower and then upper bound, and then subjects needs to just be the subject ids, no list brackets.

### Decoding
1. run ```sbatch sbatch_decoding_dcc.sh``` from the dcc_scripts/decoding folder on the dcc, with chosen parameters in run_decoding_dcc.py. Testing code is at bottom of this file, uncomment to test. Make sure your chosen epochs root file is saved in the dcc cwork folder. Using unit of analysis as repeat right now. The decoding_dcc.py script will load in the epoched data of specified subjects, then for each bootstrap, transform it into a LabeledArray where each electrode is randomly downsampled to the lowest number of trials across electrodes in that roi and condition, then for each condition comparison (i.e., congruent vs incongruent), randomly downsample again to the lowest number of trials across conditions for that condition comparison, then run decoding where error bars and stats are calculated using the unit of analysis (bootstrap, repeat, fold). If bootstrap, it will sum accuracies across folds and average across repeats for each bootstrap. If repeat, it will sum accuracies across folds for each repeat. If fold, it will find the variance and stats over all folds.
### RSA
1. rsa.ipynb uses my math to do RSA. rsa_using_toolbox.ipynb uses the rsatoolbox library (and also does power trace plotting too).
 
### Steps for new subjects
1. Run plot_clean.ipynb to preprocess (line noise filtering) for new subjects
2. Get significant electrodes by running plot_hg_and_stats with Stimulus as the event (should be top cell I think). Also run the bottom cells to plot individual electrodes for raw traces and high gamma filtered in this step.
3. Get high gamma of main effect conditions by running plot_hg_and_stats with events as Stimulus/i25 + Stimulus/i75 (inc), and Stimulus/c25 + Stimulus/c75 (con), and Stimulus/i25/s25 + Stimulus/i25/s75 + Stimulus/i75/s25 + Stimulus/i75/s75 + Stimulus/c25/s25 + Stimulus/c25/s75 + Stimulus/c75/s25 + Stimulus/c75/s75 (switch), and Stimulus/i25/r25 + Stimulus/i25/r75 + Stimulus/i75/r25 + Stimulus/i75/r75 + Stimulus/c25/r25 + Stimulus/c25/r75 + Stimulus/c75/r25 + Stimulus/c75/r75 (repeat)
5. Run roi_analysis.ipynb to get condition plots and stats for rois of interest. Choose rois at top of script. Note that the structure used in plot_subjects currently uses all electrodes from all selected rois, so if just want to plot one roi, just select that one.
6. run whole_brain_analysis.ipynb to get condition plots and stats for all electrodes.
7. Run plot_subjects.ipynb to get brain plots for new subjects
8. Run copy_wavelet_spec.ipynb and then plot_wavelets.ipynb for new subjects with Stimulus and Response as the events

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

Stimulus Interval (stimInterval) = 2.5 seconds.

- How long stimulus is on the screen and participant is able to respond. Increasing this time (to 2, or 2.5s at most) will (probably) make the task easier, though this hasn’t been confirmed. The task will automatically proceed to the next screen (feedback) as soon as the participant responds.

Fixation Cross Interval (fixInterval) = 0.5 seconds

- If the trial is the first of a section (or the first after pausing the task), the fixation interval is 2 seconds longer than fixIinterval so the participant has time to prepare.

Number of Blocks (numBlocks) = 4

- This should never be changed due to study design.

Trials Per Block in Main Task (trialsPerBlock) = 112

- This can be changed by increments of 16 trials. With a stiminterval of 1.5s, fixInterval of 0.5 seconds, trials run at longest for  3.4 seconds, so 128 trials x 4 blocks x 3.4 seconds = 29 minutes time spent performing main task (not including block breaks or instructions). This can be considerably shorter however, since trials proceed automatically when the participant responds. Assuming an average RT of ~1000 ms, experiment run time (not including breaks and instructions) = ~25 minutes.

Practice Accuracy Cutoff (pracAccCutoff) = 75

- Minimum accuracy percent (out of 100) needed to move on from one practice task to the next.
  
blockTypes  
A: 25% congruent, 25% switch  
B: 25% congruent, 75% switch  
C: 75% congruent, 25% switch  
D: 75% congruent, 75% switch
  
BIDS EVENTS ARE SAVED IN TERMS OF INCONGRUENT PROPORTION, NOT CONGRUENT PROPORTION. So Stimulus/c25/s25 will grab the stimulus onsets of trials that are congruent and switch, in a 25% incongruent and 25% switch block. AKA a 75% congruent and 25% switch block.
