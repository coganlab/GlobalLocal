function exited = mainTask(windowPtr, fontDefault, maxWidth, maxHeight, blockOrder, colorMapping, taskMapping, ...
fixInterval, stimInterval, trialsPerBlock, subID, subjectDir, ifi, completedMainBlocks)

    %% Set up prelim variables
    Screen('TextStyle',windowPtr, 0);
    c = clock; %Current date and time as date vector. [year month day hour minute seconds]
    time =strcat(num2str(c(1)),'_',num2str(c(2)),'_',num2str(c(3)),'_',num2str(c(4)),'_',num2str(c(5))); %makes unique filename
    exited = 0;
    
    % allowable responses
    ans_left = KbName('LeftArrow'); % KbName('LeftArrow'); %left response
    ans_right = KbName('RightArrow'); % KbName('RightArrow'); %right response
    escape = KbName('ESCAPE');

    % set block order
    if (blockOrder == 1)
        mainTask.blockOrder = {'A','B','D','C'};
    elseif (blockOrder == 2)
        mainTask.blockOrder = {'B','C','A','D'};
    elseif (blockOrder == 3)
        mainTask.blockOrder = {'C','D','B','A'};
    else
        mainTask.blockOrder = {'D','A','C','B'};
    end
    
    %determine which blocks have been run already or still need
    if numel(completedMainBlocks) > 0
        if numel(completedMainBlocks) < 4
            blocksToComplete = mainTask.blockOrder(numel(completedMainBlocks) + 1:end);
        else
            blocksToComplete = mainTask.blockOrder;
        end
    else
        blocksToComplete = mainTask.blockOrder;
    end
    
    %Draw S and H letters to get pixel sizes (for drawing navon stimuli)
    Screen(windowPtr,'TextSize',fontDefault);
    Screen('TextStyle',windowPtr, 0);
    [sx, sy, sBox] = DrawFormattedText(windowPtr, 'S', 'center', 'center', [0 0 0]);
    sWidth = sBox(3) - sBox(1);
    sHeight = sBox(4) - sBox(2);
    [hx, hy, hBox] = DrawFormattedText(windowPtr, 'H', 'center', 'center', [0 0 0]);
    hWidth = hBox(3) - hBox(1);
    hHeight = hBox(4) - hBox(2);
    %cover with white so participant doesnt see
    Screen('FillRect', windowPtr, [255, 255, 255]);
    
    %Double Flip to clear screen after drawing and measuring
    Screen('Flip',windowPtr);
    Screen('Flip',windowPtr);
    
    %stimulus flip time information
    %fixFlipFrames = round(fixInterval / ifi);
    stimFlipFrames = round(stimInterval/ifi);
    
    % Circle stuff for photodiode
    baseCircleDiam = 75; 
    baseCircle = [0 0 baseCircleDiam baseCircleDiam];
    centeredCircle = CenterRectOnPointd(baseCircle, maxWidth-0.5*baseCircleDiam, 1+0.5*baseCircleDiam); %
    circleColor1 = [1 1 1]; % white
    circleColor2 = [0 0 0]; % black
    
    %% create experiment trials variables for each block (see functions at bottom)
    for i = 1:numel(mainTask.blockOrder)
        blockType = char(mainTask.blockOrder(i));
        
        %create target array (is correct answer 'S' or 'H')
        targetArr = createTargetArr(trialsPerBlock);
        mainTask.blocks.(['Block_' int2str(i)]).targetArr = targetArr;
        
        %create task array (global vs local)
        taskArr = createTaskArr(trialsPerBlock, blockType);
        mainTask.blocks.(['Block_' int2str(i)]).taskArr = taskArr;
        
        %create congruency array (is trial congruent or incongruent)
        congruencyArr = createCongruencyArr(trialsPerBlock, blockType);
        mainTask.blocks.(['Block_' int2str(i)]).congArr = congruencyArr;
        
        %create stimulus array (given target, task, and congruency, what is the
        %appropriate navon stimulus?)
        mainTask.blocks.(['Block_' int2str(i)]).stimStruct = createStimulusStruct(targetArr, taskArr, congruencyArr);
        
        %create array with which action (response) is correct 
        mainTask.blocks.(['Block_' int2str(i)]).actionArr = createActionArr(taskMapping,targetArr);
        
        %create array of if trial is switch or repeat
        mainTask.blocks.(['Block_' int2str(i)]).switchTypeArr = createSwitchTypeArr(taskArr);
        
    end
    
    %% Main Task Loop
    %Get timing information
    taskStartTime = GetSecs;
    accCount = 0;
    trialCount = 1;
    rowIterator = 1;
    withinRunBlockN = 1;
    for block = 1:numel(mainTask.blockOrder) %for each block in all possible blocks
        
        %get block info
        blockType = char(mainTask.blockOrder(block));
        blockStartTime = GetSecs;
        
        % see if block needs to be run or not
        if any(strcmp(blocksToComplete, blockType))
            
            paused = 0;
            trial = 1;
            while trial <= trialsPerBlock %trial loop

                % get fixation interval
                if trial == 1 || paused
                    fix_int = fixInterval + 2; %slightly longer initial fixation if after pause or first trial
                else
                    fix_int = fixInterval;
                end

                %reset paused vars
                paused = 0;

                %fixation screen
                fixation();
                fixStart = Screen('Flip',windowPtr);
                fixOnset = fixStart - taskStartTime;
                while GetSecs - fixStart < fix_int && ~paused
                    [keyPressed,~,keyCode] = KbCheck;
                    if (keyCode(escape) == 1)
                        paused = 1;
                    end   
                end
                fixEnd = GetSecs;

                if paused 

                    %paused during cue interval
                    exited = pause();
                    if exited
                        return;
                    end

                else

                    % prepare for stimulus presentation
                    stim = mainTask.blocks.(['Block_' int2str(block)]).stimStruct.(['trial' int2str(trial)]);
                    flipTimes = zeros(1,stimFlipFrames);
                    acc = 0;
                    RT = NaN;
                    responseKey = NaN;
                    respOnset = NaN;
                    frameCount = 1;

                    %display stimulus and wait for response
                    keyPressed = 0;
                    while ~keyPressed && frameCount <= stimFlipFrames
                        [keyPressed,respOnset,keyCode] = KbCheck;

                        if frameCount <= 3
                            %photodiode for first 3 frames
                            Screen('FillOval', windowPtr, circleColor1, centeredCircle, baseCircleDiam);
                        end

                        %draw stimulus and frame
                        drawStimulus(stim);
                        drawRect(char(mainTask.blocks.(['Block_' int2str(block)]).taskArr(trial)));
                        drawReminder();

                        %flip screen
                        flipTimes(1,frameCount) = Screen('Flip',windowPtr);

                        %update frame counter
                        frameCount = frameCount + 1;
                    end

                    %get timings
                    stimStart = flipTimes(1,1);
                    stimOnset = stimStart - taskStartTime;
                    stimEnd = GetSecs;

                    %determine accuracy after response
                    if keyPressed
                        if (keyCode(ans_left) == 1)
                            acc = mainTask.blocks.(['Block_' int2str(block)]).actionArr(trial) == ans_left;
                        elseif (keyCode(ans_right) == 1)    
                            acc = mainTask.blocks.(['Block_' int2str(block)]).actionArr(trial) == ans_right;
                        elseif (keyCode(escape) == 1)
                            paused = 1;
                        end
                    end

                    if paused

                        %paused during stim window
                        exited = pause();
                        if exited
                            return;
                        end

                    else

                        %show feedback based on accuracy or if they responded at all
                        if keyPressed == 0
                            partResponded = 0;
                            feedback(acc, 1);
                        else
                            responseKey = find(keyCode==1);
                            RT = round((respOnset-stimStart)*1000);
                            partResponded = 1;
                            feedback(acc, 0);
                            if acc == 1
                                accCount = accCount + 1;
                            end
                        end

                        %feedback delay
                        feedbackTime = Screen('Flip',windowPtr);
                        while GetSecs - feedbackTime < ITIInterval && ~paused
                            [~,~,keyCode] = KbCheck;
                            if (keyCode(escape) == 1)
                                paused = 1;
                            end   
                        end

                        %Log and save data
                        if isempty(responseKey)
                            respOnset = NaN;
                            responseKey = NaN;
                        end

                        %log data
                        mainTask.taskData.logType{rowIterator,1} = "task";
                        mainTask.taskData.withinRunBlockN{rowIterator,1} = withinRunBlockN;
                        mainTask.taskData.absBlockN{rowIterator,1} = block;
                        mainTask.taskData.blockType{rowIterator,1} = mainTask.blockOrder(block);
                        mainTask.taskData.trialCount{rowIterator,1} = trialCount;
                        mainTask.taskData.blockTrialCount{rowIterator,1} = trial;
                        mainTask.taskData.fixStart{rowIterator,1} = fixStart;
                        mainTask.taskData.fixEnd{rowIterator,1} = fixEnd;
                        mainTask.taskData.fixOnset{rowIterator,1} = fixOnset;
                        mainTask.taskData.blockTrialCount{rowIterator,1} = trial;
                        mainTask.taskData.partResponded{rowIterator,1} = partResponded;
                        mainTask.taskData.partResp{rowIterator,1} = responseKey; %what button was pressed
                        mainTask.taskData.corrResp{rowIterator,1} = mainTask.blocks.(['Block_' int2str(block)]).actionArr(trial); %what was correct answer
                        mainTask.taskData.acc{rowIterator,1} = acc; %acc
                        mainTask.taskData.globalStim{rowIterator,1} = mainTask.blocks.(['Block_' int2str(block)]).stimStruct.(['trial' int2str(trial)]).global;
                        mainTask.taskData.localStim{rowIterator,1} = mainTask.blocks.(['Block_' int2str(block)]).stimStruct.(['trial' int2str(trial)]).local;
                        mainTask.taskData.stimOnset{rowIterator,1} = stimOnset; %time stim was presented (since start of main task)
                        mainTask.taskData.respOnset{rowIterator,1} = respOnset - taskStartTime; %time key was pressed (if pressed, else NaN);
                        mainTask.taskData.RT{rowIterator,1} = RT; %stimOnset - respOnset
                        mainTask.taskData.stimStart {rowIterator,1} = stimStart;
                        mainTask.taskData.stimEnd {rowIterator,1} = stimEnd;
                        mainTask.taskData.congruency{rowIterator,1} = mainTask.blocks.(['Block_' int2str(block)]).congArr(trial); %congruency of stim
                        mainTask.taskData.task{rowIterator,1} = mainTask.blocks.(['Block_' int2str(block)]).taskArr(trial); %what task was being done
                        mainTask.taskData.target{rowIterator,1} = mainTask.blocks.(['Block_' int2str(block)]).targetArr(trial); %what was the letter for correct answer
                        mainTask.taskData.switchType{rowIterator,1} = mainTask.blocks.(['Block_' int2str(block)]).switchTypeArr(trial); %what was switchtype of trial
                        mainTask.taskData.subject_ID{rowIterator,1} = convertCharsToStrings(subID);
                        mainTask.taskData.pauseDuration{rowIterator,1} = NaN;
                        rowIterator = rowIterator + 1; %iterate so next data goes into next line

                        %save data every  trial (in case of crash)
                        out_data = struct2table(mainTask.taskData);
                        writetable(out_data,strcat(subjectDir,'/GL_MainTask_Data_',subID,'_',time,'.csv'));

                        %if gotten to this point, advance trial, even if task is
                        %paused during feedback, and log data
                        trial = trial + 1;
                        trialCount = trialCount + 1;

                        %pause or advance
                        if paused

                            exited = pause();
                            if exited
                                return;
                            end

                        end %if paused during feedback

                    end %if paused during stim

                end %if paused during fix

            end %trial loop

            %save out that block was completed
            completedMainBlocks(end + 1) = {mainTask.blockOrder(block)};
            save(['data/' subID '/completedMainBlocks.mat'],'completedMainBlocks');

            % post block screens
            if withinRunBlockN < numel(blocksToComplete) %if runs remain
                Screen(windowPtr,'TextSize',36);
                Screen('TextStyle',windowPtr, 0);
                DrawFormattedText(windowPtr, ['You have reached the end of block ' num2str(withinRunBlockN) '.\n\n You have ' ...
                    num2str(numel(blocksToComplete) - withinRunBlockN) ' blocks remaining.\n\n\n Press any button to begin the next block.'], 'center', 'center', [0 0 0]);
                Screen('Flip',windowPtr);
                WaitSecs(0.5);

                %indicate whether to continue or the exit the experiment
                keyPressed = 0;
                while ~keyPressed
                    [keyPressed,secs,keyCode] = KbCheck;
                    if (keyCode(escape) == 1)
                        exited = 1;
                    end  
                end

                if exited
                    %are you sure you want to exit?
                    DrawFormattedText(windowPtr, 'Are you sure you want to exit?', 'center', maxHeight / 2 - 100, [0 0 0], 100);
                    DrawFormattedText(windowPtr, 'Press esc to confirm exit.', 'center', maxHeight / 2 + 100, [0 0 0], 100);
                    DrawFormattedText(windowPtr, 'Press any other button to proceed to the next block', 'center', maxHeight / 2 + 200, [0 0 0], 100);
                    Screen('Flip',windowPtr);
                    WaitSecs(0.5);

                    %wait for response
                    keyPressed = 0;
                    exited = 0;
                    while ~keyPressed
                        [keyPressed,secs,keyCode] = KbCheck;
                        if (keyCode(escape) == 1)
                            exited = 1;
                        end  
                    end

                    if exited == 1
                        return;
                    end
                end
            end %post block feedback screens
            
            %increment within run block counter
            withinRunBlockN = withinRunBlockN + 1;
            
        end %if block needs to be run
        
    end %block loop
    
    %% End of Experiment
    
    %% experiment functions
    
    function fixation()
        %Change font size
        Screen(windowPtr,'TextSize',fontDefault);
        Screen('TextStyle',windowPtr, 0);

        %Draw fixation
        DrawFormattedText(windowPtr, '+', 'center', maxHeight / 2, [0 0 0]);
    end
    
    function drawStimulus(stim)
        %Change Font Size
        Screen(windowPtr,'TextSize',fontDefault);
        Screen('TextStyle',windowPtr, 0);

        if strcmp(stim.global, 's')
            if strcmp(stim.local, 's')
                %S made of S
                DrawFormattedText(windowPtr, 'SSS', 'center', maxHeight / 2 - 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) - (3 * sWidth / 2), maxHeight / 2 - 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'SSS', 'center', maxHeight / 2, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) + (sWidth / 2), maxHeight / 2 + 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'SSS', 'center', maxHeight / 2 + 120, [0 0 0]);
            else
                %S made of H
                DrawFormattedText(windowPtr, 'HHH', 'center', maxHeight / 2 - 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) - (3 * hWidth / 2), maxHeight / 2 - 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'HHH', 'center', maxHeight / 2, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) + (hWidth / 2), maxHeight / 2 + 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'HHH', 'center', maxHeight / 2 + 120, [0 0 0]);
            end
        else
            if strcmp(stim.local, 's')
                %H made of S
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) - (3 * sWidth / 2), maxHeight / 2 - 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) + (sWidth / 2), maxHeight / 2 - 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) - (3 * sWidth / 2), maxHeight / 2 - 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) + (sWidth / 2), maxHeight / 2 - 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'SSS', 'center', maxHeight / 2, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) - (3 * sWidth / 2), maxHeight / 2 + 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) + (sWidth / 2), maxHeight / 2 + 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) - (3 * sWidth / 2), maxHeight / 2 + 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'S', (maxWidth/2) + (sWidth / 2), maxHeight / 2 + 120, [0 0 0]);

            else
                %H made of H
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) - (3 * hWidth / 2), maxHeight / 2 - 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) + (hWidth / 2), maxHeight / 2 - 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) - (3 * hWidth / 2), maxHeight / 2 - 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) + (hWidth / 2), maxHeight / 2 - 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'HHH', 'center', maxHeight / 2, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) - (3 * hWidth / 2), maxHeight / 2 + 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) + (hWidth / 2), maxHeight / 2 + 60, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) - (3 * hWidth / 2), maxHeight / 2 + 120, [0 0 0]);
                DrawFormattedText(windowPtr, 'H', (maxWidth/2) + (hWidth / 2), maxHeight / 2 + 120, [0 0 0]);
            end
        end
    end

    function drawRect(task)
        % draw rectangle based on cue and color mapping counterbalacing
        if strcmp(task, 'g')
            if (colorMapping == 1)
                Screen('FrameRect',windowPtr, [255, 0, 0],[maxWidth/2 - 110, maxHeight/2 - 200, maxWidth/2 + 110, maxHeight/2 + 150], 6);
            else
                Screen('FrameRect',windowPtr, [0,0,255],[maxWidth/2 - 110, maxHeight/2 - 200, maxWidth/2 + 110, maxHeight/2 + 150], 6);
            end
        else
            if (colorMapping == 1)
                Screen('FrameRect',windowPtr, [0, 0, 255],[maxWidth/2 - 110, maxHeight/2 - 200, maxWidth/2 + 110, maxHeight/2 + 150], 6);
            else
                Screen('FrameRect',windowPtr, [255,0,0],[maxWidth/2 - 110, maxHeight/2 - 200, maxWidth/2 + 110, maxHeight/2 + 150], 6);
            end
        end
        
    end

    function drawReminder()
        Screen('TextStyle',windowPtr, 0);
        Screen(windowPtr,'TextSize',24);
        if colorMapping == 1

            Screen('FillRect', windowPtr, [255, 0, 0],[maxWidth/2 - 140, maxHeight/2 + 315, maxWidth/2 - 100, maxHeight/2 + 355]);
%            DrawFormattedText(windowPtr, 'Global', maxWidth/2 - 90, maxHeight / 2 + 345, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Big', maxWidth/2 - 90, maxHeight / 2 + 345, [0 0 0], 100);
            
            
            Screen('FillRect', windowPtr, [0, 0, 255],[maxWidth/2 + 30, maxHeight/2 + 315, maxWidth/2 + 70, maxHeight/2 + 355]);
%            DrawFormattedText(windowPtr, 'Local', maxWidth/2 + 80, maxHeight / 2 + 345, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Small', maxWidth/2 + 80, maxHeight / 2 + 345, [0 0 0], 100);
            
        else
            
            Screen('FillRect', windowPtr, [0, 0, 255],[maxWidth/2 - 140, maxHeight/2 + 315, maxWidth/2 - 100, maxHeight/2 + 355]);
%            DrawFormattedText(windowPtr, 'Global', maxWidth/2 - 90, maxHeight / 2 + 345, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Big', maxWidth/2 - 90, maxHeight / 2 + 345, [0 0 0], 100);
            
            
            Screen('FillRect', windowPtr, [255, 0, 0],[maxWidth/2 + 30, maxHeight/2 + 315, maxWidth/2 + 70, maxHeight/2 + 355]);
%            DrawFormattedText(windowPtr, 'Local', maxWidth/2 + 80, maxHeight / 2 + 345, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Small', maxWidth/2 + 80, maxHeight / 2 + 345, [0 0 0], 100);

        end
    end

    function feedback(acc, tooSlow)
        Screen(windowPtr,'TextSize',fontDefault);
        if (tooSlow == 1)
            DrawFormattedText(windowPtr, 'Too Slow!', 'center', maxHeight / 2, [0,0,0]);
        elseif (acc == 1)
            DrawFormattedText(windowPtr, 'Correct', 'center', maxHeight / 2, [0,128,0]);
        else
            DrawFormattedText(windowPtr, 'Incorrect', 'center', maxHeight / 2, [255,0,0]);
        end
    end

    function exited = pause()

        Screen(windowPtr,'TextSize',32);
        Screen('TextStyle',windowPtr, 0);
        instruction = char("You paused the task.");
        DrawFormattedText(windowPtr, instruction, 'center',  maxHeight / 2 - 100, [0 0 0], 100);
        instruction = char("Press any button to resume, or escape to exit the experiment.");
        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 + 100, [0 0 0], 100);
        pauseTime = Screen('Flip',windowPtr);

        WaitSecs(0.5);
        keyPressed = 0;
        while ~keyPressed
            [keyPressed,secs,keyCode] = KbCheck;
            if keyCode(escape) == 1
                exited = 1;
            else
                exited = 0;
            end
        end
        
        pauseDuration = GetSecs - pauseTime;
        %log data that pause happened, and for how long
        mainTask.taskData.logType{rowIterator,1} = "pause";
        mainTask.taskData.withinRunBlockN{rowIterator,1} = withinRunBlockN;
        mainTask.taskData.absBlockN{rowIterator,1} = block;
        mainTask.taskData.blockType{rowIterator,1} = mainTask.blockOrder(block);
        mainTask.taskData.trialCount{rowIterator,1} = NaN;
        mainTask.taskData.blockTrialCount{rowIterator,1} = NaN;
        mainTask.taskData.fixStart{rowIterator,1} = NaN;
        mainTask.taskData.fixEnd{rowIterator,1} = NaN;
        mainTask.taskData.fixOnset{rowIterator,1} = NaN;
        mainTask.taskData.blockTrialCount{rowIterator,1} = NaN;
        mainTask.taskData.partResponded{rowIterator,1} = NaN;
        mainTask.taskData.partResp{rowIterator,1} = NaN;
        mainTask.taskData.corrResp{rowIterator,1} = NaN;
        mainTask.taskData.acc{rowIterator,1} = NaN;
        mainTask.taskData.globalStim{rowIterator,1} = NaN;
        mainTask.taskData.localStim{rowIterator,1} = NaN;
        mainTask.taskData.stimOnset{rowIterator,1} = NaN;
        mainTask.taskData.respOnset{rowIterator,1} = NaN;
        mainTask.taskData.RT{rowIterator,1} = NaN;
        mainTask.taskData.stimStart {rowIterator,1} = NaN;
        mainTask.taskData.stimEnd {rowIterator,1} = NaN;
        mainTask.taskData.congruency{rowIterator,1} = NaN;
        mainTask.taskData.task{rowIterator,1} = NaN;
        mainTask.taskData.target{rowIterator,1} = NaN;
        mainTask.taskData.switchType{rowIterator,1} = NaN;
        mainTask.taskData.subject_ID{rowIterator,1} = convertCharsToStrings(subID);
        mainTask.taskData.pauseDuration{rowIterator,1} = pauseDuration;
        rowIterator = rowIterator + 1; %iterate so next data goes into next line
    end
end

%% Experiment functions

function interval = ITIInterval()
    min = 1200; %1200
    max = 1400; %1400
    step = 50;
    nSteps = floor((max - min) / step);
    interval = (min + randi([0 nSteps]) * step) / 1000;
end

function targetArr = createTargetArr(nTrials)
    batchSize = 16;
    targetArr = {};
    for i = 1:ceil(nTrials/batchSize)
        newBatch = createTargetBatch(batchSize);
        targetArr = cat(1,targetArr,newBatch);
    end
end

function targetBatch = createTargetBatch(batchSize)
    % create array of 'S' trial type
    sArr = cell(floor(batchSize / 2), 1);
    for k=1:length(sArr)
        sArr(k) = {'s'};
    end
    
    % create array of 'H' trial type
    hArr = cell(batchSize - length(sArr), 1);
    for k=1:length(hArr)
        hArr(k) = {'h'};
    end
    
    targArr = shuffle(cat(1,hArr,sArr));
    
    while (~targArrOk(targArr))
        targArr = shuffle(targArr);
    end
    
    targetBatch = targArr;
    
    function isOk = targArrOk(arr)
        nRepeats = 0;
        
        for i = 1:length(arr)
            if i < length(arr)
                if strcmp(arr(i),arr(i+1))
                    nRepeats = nRepeats + 1;
                end
            end
        end
        
        isOk = ~(nRepeats < 7 || nRepeats > 9);
    end
end

function taskArr = createTaskArr(blockSize, blockLetter)
    taskA = 'g';
    taskB = 'l';
    
    %determine blockwise switch proportion
    if (blockLetter == 'A' || blockLetter == 'C')
        percSwitch = 0.25;
    else
        percSwitch = 0.75;
    end
    
    %get trial numbers
    nSwitchTrials = floor(blockSize * percSwitch);
    nRepeatTrials = blockSize - nSwitchTrials;
    
    %randomly pick first trial of batch
    if (rand < 0.5)
        task1 = taskA;
        task2 = taskB;
    else
        task1 = taskB;
        task2 = taskA;
    end
    
    %create just switches array
    arr = cell(1, nSwitchTrials);
    for i = 1:length(arr)
        if (rem(i,2) == 1)
            arr(i) = {task1};
        else
            arr(i) = {task2};
        end
    end
    
    %insert repeat trials into switches arr
    for i = 1:ceil(nRepeatTrials/2)
        a_indices = getAllIndices(arr,taskA);
        loc_A = datasample(a_indices,1);
        arr = [arr(1:loc_A),taskA, arr(loc_A + 1:end)];
        
        b_indices = getAllIndices(arr,taskB);
        loc_B = datasample(b_indices,1);
        arr = [arr(1:loc_B),taskB, arr(loc_B + 1:end)];
    end
    
    taskArr = transpose(arr);
    
    function indices = getAllIndices(arr, val)
        indices = [];
        for j = 1:length(arr)
            if strcmp(arr(j),val)
                indices = [indices, j];
            end
        end
        
    end
    
end

function congruencyArr = createCongruencyArr(blockSize, blockLetter)
    if (blockLetter == 'A' || blockLetter == 'B')
        percInc = 0.75;
    else
        percInc = 0.25;
    end

    % create array with incongruent stim
    incArr = cell(floor(blockSize * percInc), 1);
    for k=1:length(incArr)
        incArr(k) = {'i'};
    end
    
    %create arr with congruent stim
    conArr = cell(blockSize - length(incArr), 1);
    for k=1:length(conArr)
        conArr(k) = {'c'};
    end
    
    %concat and return shuffled
    congruencyArr = shuffle(cat(1,incArr,conArr));
end

function stimStruct = createStimulusStruct(targetArr, taskArr, congruencyArr)

    for i = 1:length(targetArr)
        target = char(targetArr(i)); 
        if (strcmp(congruencyArr(i),'c'))
            trials.(['trial' num2str(i)]).global = target;
            trials.(['trial' num2str(i)]).local = target;
        else
            if strcmp(taskArr(i),'g')
                trials.(['trial' num2str(i)]).global = target;
                trials.(['trial' num2str(i)]).local = oppositeOfTarget(target);
            else
                trials.(['trial' num2str(i)]).global = oppositeOfTarget(target);
                trials.(['trial' num2str(i)]).local = target;
            end
        end
    end
    %return finished stim struct
    stimStruct = trials;

end

function switchTypeArr = createSwitchTypeArr(taskArr)

    arr = cell(length(taskArr), 1);
    for i = 1:length(taskArr)
        if i == 1
            arr(i) = {'n'};
        elseif strcmp(char(taskArr(i)),char(taskArr(i-1))) %repeat trial
            arr(i) = {'r'};
        else %switch
            arr(i) = {'s'};
        end
    end
    switchTypeArr = arr;
end

function actionArr = createActionArr(taskMapping, targetArr)
    arr = zeros(length(targetArr),1);
    for i = 1:length(targetArr)
        if taskMapping == 1
            if strcmp(char(targetArr(i)),'s')
                arr(i) = KbName('LeftArrow');
            else
                arr(i) = KbName('RightArrow');
            end
        else
            if strcmp(char(targetArr(i)),'s')
                arr(i) = KbName('RightArrow');
            else
                arr(i) = KbName('LeftArrow');
            end
        end
    end
    actionArr = arr;
end

function opposite = oppositeOfTarget(target) 
    if strcmp(target,'s')
        opposite = 'h';
    else
        opposite = 's';
    end
end

function v=shuffle(v)
     v=v(randperm(length(v)));
end
 
