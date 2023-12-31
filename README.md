﻿Global Local Task 

Contact: Raphael Geddert

raphael.geddert@duke.edu

Last edited: 02-18-2021

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

