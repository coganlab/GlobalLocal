function [accuracy, exited] = practiceLocal(windowPtr, fontDefault, maxWidth, maxHeight, ...
colorMapping, taskMapping, nTrials, iteration, fixInterval, stimInterval, subID, subjectDir, ifi)

    %% Set up prelim vars
    Screen('TextStyle',windowPtr, 0);
    accuracy = 0;
    paused = 0;
    c = clock; %Current date and time as date vector. [year month day hour minute seconds]
    time =strcat(num2str(c(1)),'_',num2str(c(2)),'_',num2str(c(3)),'_',num2str(c(4)),'_',num2str(c(5))); %makes unique filename
    exited = 0;
    
    % allowable responses
    ans_left = KbName('LeftArrow'); % KbName('LeftArrow'); %left response
    ans_right = KbName('RightArrow'); % KbName('RightArrow'); %right response
    escape = KbName('ESCAPE');
    
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
    stimFlipFrames = round(stimInterval/ifi);
    
    % Circle stuff for photodiode
    baseCircleDiam = 75; 
    baseCircle = [0 0 baseCircleDiam baseCircleDiam];
    centeredCircle = CenterRectOnPointd(baseCircle, maxWidth-0.5*baseCircleDiam, 1+0.5*baseCircleDiam); %
    circleColor1 = [1 1 1]; % white
    circleColor2 = [0 0 0]; % black
    
    %% Create task arrays
    targetArr = createTargetArr(nTrials);
    taskArr = createTaskArr(nTrials, 'l');
    congruencyArr = createCongruencyArr(nTrials);
    stimStruct = createStimulusStruct(targetArr, taskArr, congruencyArr);
    actionArr = createActionArr(taskMapping,targetArr);
    switchTypeArr = createSwitchTypeArr(taskArr);
    
    %% Experiment loop
    accCount = 0;
    taskStartTime = GetSecs;
    trial = 1;
    rowIterator = 1;
    while trial <= nTrials
        
        % get fixation interval
        if trial == 1 || paused
            fix_int = fixInterval + 2; %slightly longer initial fixation if after pause or first trial
        else
            fix_int = fixInterval;
        end
        
        %reset paused vars
        paused = 0;
        
        %fixation delay
        fixation();
        fixStart = Screen('Flip',windowPtr);
        fixOnset = fixStart - taskStartTime;
        while GetSecs - fixStart < fix_int && ~paused
            [keyPressed,respOnset,keyCode] = KbCheck;
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
            stim = stimStruct.(['trial' int2str(trial)]);
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
                drawRect(char(taskArr(trial)));
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
                    acc = actionArr(trial) == ans_left;
                elseif (keyCode(ans_right) == 1)    
                    acc = actionArr(trial) == ans_right;
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
                    [keyPressed,respOnset,keyCode] = KbCheck;
                    if (keyCode(escape) == 1)
                        paused = 1;
                    end   
                end
                
                %log data
                pracTaskData.logType{rowIterator,1} = "task";
                pracTaskData.iteration{rowIterator,1} = iteration;
                pracTaskData.trialCount{rowIterator,1} = trial;
                pracTaskData.fixStart{rowIterator,1} = fixStart;
                pracTaskData.fixEnd{rowIterator,1} = fixEnd;
                pracTaskData.fixOnset{rowIterator,1} = fixOnset;
                pracTaskData.partResponded{rowIterator,1} = partResponded;
                pracTaskData.partResp{rowIterator,1} = responseKey;
                pracTaskData.corrResp{rowIterator,1} = actionArr(trial);
                pracTaskData.acc{rowIterator,1} = acc;
                pracTaskData.globalStim{rowIterator,1} = stimStruct.(['trial' num2str(trial)]).global;
                pracTaskData.localStim{rowIterator,1} = stimStruct.(['trial' num2str(trial)]).local;
                pracTaskData.stimOnset{rowIterator,1} = stimOnset;
                pracTaskData.respOnset{rowIterator,1} = respOnset - taskStartTime;
                pracTaskData.stimStart{rowIterator,1} = stimStart;
                pracTaskData.stimEnd{rowIterator,1} = stimEnd;
                pracTaskData.RT{rowIterator,1} = RT;
                pracTaskData.congruency{rowIterator,1} = congruencyArr(trial);
                pracTaskData.task{rowIterator,1} = taskArr(trial);
                pracTaskData.target{rowIterator,1} = targetArr(trial);
                pracTaskData.switchType{rowIterator,1} = switchTypeArr(trial);
                pracTaskData.subject_ID{rowIterator,1} = convertCharsToStrings(subID);
                pracTaskData.pauseDuration{rowIterator,1} = NaN;
                rowIterator = rowIterator + 1; %iterate so next data goes into next line
                
                %save data every trial (in case of crash)
                out_data = struct2table(pracTaskData);
                writetable(out_data,strcat(subjectDir,'/GL_Local_Practice_Data_#',num2str(iteration),'_',subID,'_',time,'.csv'));
                
                %if gotten to this point, advance trial, even if task is
                %paused during feedback
                trial = trial + 1;
                
                %pause or advance
                if paused
                    
                    exited = pause();
                    if exited
                        return;
                    end
                    
                end
                
            end
            
        end
    end
    
    %% End of task
    %save out data
    
    %calculate overall accuracy and return to main.m
    accuracy = accCount / nTrials;
    
    %% experiment functions
    function fixation()
        %Change font size
        Screen(windowPtr,'TextSize',fontDefault);
        Screen('TextStyle',windowPtr, 0);

        %Draw fixation
        [nx, ny, bbox] = DrawFormattedText(windowPtr, '+', 'center', maxHeight / 2, [0 0 0]);
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
        Screen(windowPtr,'TextSize',32);
        if colorMapping == 1

            Screen('FillRect', windowPtr, [0, 0, 255],[maxWidth/2 - 60, maxHeight/2 + 315, maxWidth/2 - 20, maxHeight/2 + 355]);
            %DrawFormattedText(windowPtr, 'Local', maxWidth/2 - 10, maxHeight / 2 + 345, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Small', maxWidth/2 - 10, maxHeight / 2 + 345, [0 0 0], 100);

        else

            Screen('FillRect', windowPtr, [255, 0, 0],[maxWidth/2 - 60, maxHeight/2 + 315, maxWidth/2 - 20, maxHeight/2 + 355]);
           % DrawFormattedText(windowPtr, 'Local', maxWidth/2 - 10, maxHeight / 2 + 345, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Small', maxWidth/2 - 10, maxHeight / 2 + 345, [0 0 0], 100);

        end
    end

    function feedback(acc, tooSlow)
        Screen(windowPtr,'TextSize',fontDefault);
        Screen('TextStyle',windowPtr, 0);
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
        
        %log data
        pauseDuration = GetSecs - pauseTime;
        pracTaskData.logType{rowIterator,1} = "pause";
        pracTaskData.iteration{rowIterator,1} = NaN;
        pracTaskData.trialCount{rowIterator,1} = NaN;
        pracTaskData.fixStart{rowIterator,1} = NaN;
        pracTaskData.fixEnd{rowIterator,1} = NaN;
        pracTaskData.fixOnset{rowIterator,1} = NaN;
        pracTaskData.partResponded{rowIterator,1} = NaN;
        pracTaskData.partResp{rowIterator,1} = NaN;
        pracTaskData.corrResp{rowIterator,1} = NaN;
        pracTaskData.acc{rowIterator,1} = NaN;
        pracTaskData.globalStim{rowIterator,1} = NaN;
        pracTaskData.localStim{rowIterator,1} = NaN;
        pracTaskData.stimOnset{rowIterator,1} = NaN;
        pracTaskData.respOnset{rowIterator,1} = NaN;
        pracTaskData.stimStart{rowIterator,1} = NaN;
        pracTaskData.stimEnd{rowIterator,1} = NaN;
        pracTaskData.RT{rowIterator,1} = NaN;
        pracTaskData.congruency{rowIterator,1} = NaN;
        pracTaskData.task{rowIterator,1} = NaN;
        pracTaskData.target{rowIterator,1} = NaN;
        pracTaskData.switchType{rowIterator,1} = NaN;
        pracTaskData.subject_ID{rowIterator,1} = NaN;
        pracTaskData.pauseDuration{rowIterator,1} = pauseDuration;
        rowIterator = rowIterator + 1; %iterate so next data goes into next line
    end
end

function interval = ITIInterval()
    min = 1200;
    max = 1400;
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

function taskArr = createTaskArr(blockSize, task)
    arr = cell(1,blockSize);
    for i = 1:length(arr)
        arr(i) = {task};
    end
    taskArr = arr;
end

function congruencyArr = createCongruencyArr(nTrials)
    percInc = 0.50;

    % create array with incongruent stim
    incArr = cell(floor(nTrials * percInc), 1);
    for k=1:length(incArr)
        incArr(k) = {'i'};
    end
    
    %create arr with congruent stim
    conArr = cell(nTrials - length(incArr), 1);
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