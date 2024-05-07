% Clear the workspace 
sca;
close all;  
clearvars;

% disable screen sync. Incompatible with my mac (OS Catalina) but
% supposedly neccessary for accurate timings so  remove this 
% (and good luck getting it to work)
Screen('Preference', 'SkipSyncTests', 1); % <-remove me?
           
%% input subject information
prompt = {'Enter Subject ID:','Practice? (0 - Main Experiment,1 - All Practice, 2 - Just Main Practice)'};
defInput = {'001','1'};     
answer = inputdlg(prompt, 'Global Local Task', [1 50], defInput);
if isempty(answer)
    return;
end
subID = char(answer(1));
practice = str2num(char(answer(2)));

%% Set up directory (make sure file names dont overwrite)
% and load counterbalancing data if participant has run a task previously
subjectDir = ['data/' subID];

if exist(subjectDir,'dir')
    %load up counterbalancing information for this participant (whatever
    %they did the first time they ever did task)
    load(strcat(subjectDir,'/counterbalance_vars.mat'));
    
    %load file if any maintask blocks have been completed
    if exist(strcat(subjectDir, '/completedMainBlocks.mat'),'file')
        load(strcat(subjectDir, '/completedMainBlocks.mat'));
        
        %check if all 4 blocks have been completed, prompt that file will
        %be overwritten
        % ############
        if numel(completedMainBlocks) == 4
            
            answer = questdlg('It appears this subject has already completed all 4 blocks of the task. Would you like to run 4 new blocks?');
            
            switch answer
                case 'Yes'
                    completedMainBlocks = {};
                otherwise
                    return;          
            end    
        end
    else
        completedMainBlocks = {};
    end
    
    %create new folder for this participant's data
    dateTime = strcat('_',datestr(now,30));
    subjectDir = strcat(subjectDir,dateTime);
    mkdir(subjectDir)
elseif ~exist(subjectDir,'dir')
    %create initial folder for this participant
    mkdir(subjectDir)
    
    %set up counterbalancing
    pracOrder = randi([1 2]);
    % 1: Practice Global First
    % 2: Practice Local First
    
    taskMapping = randi([1 2]);
    % 1: '1' key = 'S', '0' key = 'H'
    % 2: '1' key = 'H', '0' key = 'S'
    
    colorMapping = randi([1 2]);
    % 1: Global = Red, Local = Blue
    % 2: Global = Blue, Local = Red
    
    blockOrder = randi([1 4]);
    % Latin Square Counterbalancing
    % 1: A B D C
    % 2: B C A D
    % 3: C D B A
    % 4: D A C B
    %
    % Block Proportions
    % A: 75% Inc, 25% Switch
    % B: 75% Inc, 75% Switch
    % C: 25% Inc, 25% Switch
    % D: 25% Inc, 75% Switch
    
    %save counterbalancing details in folder
    save(strcat(subjectDir, '/counterbalance_vars.mat'),'pracOrder','taskMapping','colorMapping','blockOrder');

    %variable for storing completed maintask blocks in case gets that far
    completedMainBlocks = {};
end

%% Set Up Experiment

c = clock; %Current date and time as date vector. [year month day hour minute seconds]
time = strcat(num2str(c(1)),'_',num2str(c(2)),'_',num2str(c(3)),'_',num2str(c(4)),'_',num2str(c(5))); %makes unique filename

%Set up experiment variables
stimInterval = 2.5; %1.5 or 2
fixInterval = 0.5; %0.5
numBlocks = 4; %required
trialsPerBlock = 112; %128, multiples of 16
pracTrialsPerBlock = 16;
pracAccCutoff = 75; %75

%Move subject task information into table
taskVars(1,1) = "subID:";
taskVars(1,2) = subID;
taskVars(2,1) = "stimInterval:";
taskVars(2,2) = stimInterval;
taskVars(3,1) = "fixInterval:";
taskVars(3,2) = fixInterval;
taskVars(4,1) = "numBlocks:";
taskVars(4,2) = numBlocks;
taskVars(5,1) = "trialsPerBlock:";
taskVars(5,2) = trialsPerBlock;
taskVars(6,1) = "pracTrialsPerBlock:";
taskVars(6,2) = pracTrialsPerBlock;
taskVars(7,1) = "pracAccCutoff:";
taskVars(7,2) = pracAccCutoff;
taskVars(8,1) = "pracOrder:";
taskVars(8,2) = pracOrder;
taskVars(9,1) = "taskMapping:";
taskVars(9,2) = taskMapping;
taskVars(10,1) = "colorMapping:";
taskVars(10,2) = colorMapping;
taskVars(11,1) = "blockOrder:";
taskVars(11,2) = blockOrder;
table = array2table(taskVars);

%Save table
writetable(table, strcat(subjectDir,'/',subID,'_taskdata_.txt'),'Delimiter','\t','WriteVariableNames',false);

%% Set Screen Parameters
%make sure computer has correct psychtoolbox for task
AssertOpenGL;

%select external screen if possible
screens = Screen('Screens');
dispScreen = max(screens);

% Define black and white
black = BlackIndex(dispScreen);
white = WhiteIndex(dispScreen);

%open screen and get size parameters
[windowPtr, rect] = Screen('OpenWindow',dispScreen,[255 255 255]); % [%[0,0,1024,768]
Screen('BlendFunction', windowPtr, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

% Get the size of the on screen window in pixels
[maxWidth, maxHeight] = Screen('WindowSize', windowPtr);

%get flip interval
ifi = Screen('GetFlipInterval', windowPtr);

% Get the centre coordinate of the window in pixels
[xCenter, yCenter] = RectCenter(rect);

%Set Defaults
fontDefault = 70;
Screen('TextSize',windowPtr,fontDefault);
Screen('TextFont',windowPtr,'Arial');

%cursor and keypress stuff
HideCursor;
ListenChar(2);

% see http://psychtoolbox.org/docs/MaxPriority. This was coded on mac osx
Priority(MaxPriority(windowPtr));

%Prepare key press listeners
KbName('UnifyKeyNames');
ans_left = KbName('LeftArrow'); % KbName('1!'); %left response
ans_right = KbName('RightArrow'); % KbName('0)'); %right response
space = KbName('space'); %button to advance instructions/next section
escape = KbName('ESCAPE'); %escape key (to exit experiment)
continueToMain = 0; %var to check if practice happened first (slightly changes instructions)
RestrictKeysForKbCheck([]); %ensure all keypresses are tracked

%%  Main Code

expStage = "Opener";
exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, ...
pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);

%exit the script if user hit exit
if exited
    Screen('CloseAll');
    ShowCursor;
    ListenChar(0);
    Priority(0);
    return;
end

if practice == 1
    
    % Instructions Part 1
    expStage = "Practice1";
    exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, ...
    pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);
    
    %exit the script if user hit exit
    if exited
        Screen('CloseAll');
        ShowCursor;
        ListenChar(0);
        Priority(0);
        return;
    end
    
    % Practice Part 1
    pracAcc = 0;
    prac1count = 1;
    while pracAcc < pracAccCutoff/100 || prac1count == 1
        %practice task (based on practice order)
        if pracOrder == 1
            [pracAcc, exited] = practiceGlobal(windowPtr,fontDefault, maxWidth, maxHeight, colorMapping, ...
            taskMapping, pracTrialsPerBlock, prac1count, fixInterval, stimInterval, subID, subjectDir, ifi);
        else
            [pracAcc, exited] = practiceLocal(windowPtr,fontDefault, maxWidth, maxHeight, colorMapping, ...
            taskMapping, pracTrialsPerBlock, prac1count, fixInterval, stimInterval, subID, subjectDir, ifi);
        end
        prac1count = prac1count + 1;
          
        %exit the script if user hit exit
        if exited
            Screen('CloseAll');
            ShowCursor;
            ListenChar(0);
            Priority(0);
            return;
        end

        %give feedback based on accuracy
        Screen(windowPtr,'TextSize',36);
        Screen('TextStyle',windowPtr, 0);
        accuracy = round(pracAcc * 100, 2);
        if pracAcc < pracAccCutoff/100
            text = char("Your accuracy for that block was " + accuracy + ...
                "%.\n\nRemember, you need at least " + pracAccCutoff + "% correct. \n\n\n\nPress space to try again.");
            DrawFormattedText(windowPtr, text, 'center', 'center', [0 0 0]);
        else
            text = char("Great job. Your accuracy for that block was " + accuracy + "%.\n\n\n\nPress space to continue to the next practice task.");
            DrawFormattedText(windowPtr, text , 'center', 'center', [0 0 0]);
        end

        %Wait for participant to hit space then continue to next thing,
        %else esc exits that experiment.
        Screen('Flip',windowPtr);
        WaitSecs(0.5);
        spaceHit = 0;
        exited = 0;
        while ~spaceHit && ~exited
            [keyPressed,secs,keyCode] = KbCheck;
            if keyCode(escape) == 1
                exited = 1;
            elseif keyCode(space) == 1
                spaceHit = 1;
            end
        end
                
        if exited
            %are you sure?
            DrawFormattedText(windowPtr, 'Are you sure you want to exit?', 'center', maxHeight / 2 - 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press esc to confirm exit.', 'center', maxHeight / 2 + 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press any other button to go to continue.', 'center', maxHeight / 2 + 200, [0 0 0], 100);
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
                Screen('CloseAll');
                ShowCursor;
                ListenChar(0);
                Priority(0);
                return;
            end
        end
    end

    % Instructions Part 2
    expStage = "Practice2";
    exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);

    %exit the script if user hit exit
    if exited
        Screen('CloseAll');
        ShowCursor;
        ListenChar(0);
        Priority(0);
        return;
    end
    
    % Practice Part 2
    pracAcc = 0;
    prac2count = 1;
    while pracAcc < pracAccCutoff/100 || prac2count == 1
        %practice task (based on practice order)
        if pracOrder == 1
            [pracAcc, exited] = practiceLocal(windowPtr,fontDefault, maxWidth, maxHeight, colorMapping, ...
            taskMapping, pracTrialsPerBlock, prac2count, fixInterval, stimInterval, subID, subjectDir, ifi);
        else
            [pracAcc, exited] = practiceGlobal(windowPtr,fontDefault, maxWidth, maxHeight, colorMapping, ...
            taskMapping, pracTrialsPerBlock, prac2count, fixInterval, stimInterval, subID, subjectDir, ifi);
        end
        prac2count = prac2count + 1;
        
        %exit the script if user hit exit
        if exited
            Screen('CloseAll');
            ShowCursor;
            ListenChar(0);
            Priority(0);
            return;
        end

        %give feedback based on accuracy
        Screen(windowPtr,'TextSize',36);
        Screen('TextStyle',windowPtr, 0);
        accuracy = round(pracAcc * 100, 2);
        if pracAcc < pracAccCutoff/100
            text = char("Your accuracy for that block was " + accuracy + ...
                "%.\n\nRemember, you need at least " + pracAccCutoff + "% correct. \n\n\n\nPress space to try again.");
            DrawFormattedText(windowPtr, text, 'center', 'center', [0 0 0]);
        else
            text = char("Great job. Your accuracy for that block was " + accuracy + "%.\n\n\n\nPress space to continue to the next practice task.");
            DrawFormattedText(windowPtr, text , 'center', 'center', [0 0 0]);
        end

        %Wait for participant to hit space then continue to next thing,
        %else esc exits that experiment.
        Screen('Flip',windowPtr);
        WaitSecs(0.5);
        spaceHit = 0;
        exited = 0;
        while ~spaceHit && ~exited
            [keyPressed,secs,keyCode] = KbCheck;
            if keyCode(escape) == 1
                exited = 1;
            elseif keyCode(space) == 1
                spaceHit = 1;
            end
        end
                
        if exited
            %are you sure?
            DrawFormattedText(windowPtr, 'Are you sure you want to exit?', 'center', maxHeight / 2 - 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press esc to confirm exit.', 'center', maxHeight / 2 + 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press any other button to continue.', 'center', maxHeight / 2 + 200, [0 0 0], 100);
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
                Screen('CloseAll');
                ShowCursor;
                ListenChar(0);
                Priority(0);
                return;
            end
        end
    end

    % Instructions Part 3
    expStage = "Practice3";
    exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);

    %exit the script if user hit exit
    if exited
        Screen('CloseAll');
        ShowCursor;
        ListenChar(0);
        Priority(0);
        return;
    end
    
    % Practice Part 3
    pracAcc = 0;
    prac3count = 1;
    while pracAcc < pracAccCutoff/100 || prac3count == 1
        [pracAcc, exited] = practiceGlobalLocal(windowPtr,fontDefault, maxWidth, maxHeight, colorMapping, ...
        taskMapping, pracTrialsPerBlock, prac3count, fixInterval, stimInterval, subID, subjectDir, ifi);
        prac3count = prac3count + 1;
        
        %exit the script if user hit exit
        if exited
            Screen('CloseAll');
            ShowCursor;
            ListenChar(0);
            Priority(0);
            return;
        end

        %give feedback based on accuracy
        Screen(windowPtr,'TextSize',36);
        Screen('TextStyle',windowPtr, 0);
        accuracy = round(pracAcc * 100, 2);
        if pracAcc < pracAccCutoff/100
            text = char("Your accuracy for that block was " + accuracy + ...
                "%.\n\nRemember, you need at least " + pracAccCutoff + "% correct. \n\n\n\nPress space to try again.");
            DrawFormattedText(windowPtr, text, 'center', 'center', [0 0 0]);
        else
            text = char("Great job. Your accuracy for that block was " + accuracy + "%.\n\n\n\nPress space to continue.");
            DrawFormattedText(windowPtr, text , 'center', 'center', [0 0 0]);
        end

        %Wait for participant to hit space then continue to next thing,
        %else esc exits that experiment.
        Screen('Flip',windowPtr);
        WaitSecs(0.5);
        spaceHit = 0;
        exited = 0;
        while ~spaceHit && ~exited
            [keyPressed,secs,keyCode] = KbCheck;
            if keyCode(escape) == 1
                exited = 1;
            elseif keyCode(space) == 1
                spaceHit = 1;
            end
        end
                
        if exited
            %are you sure?
            DrawFormattedText(windowPtr, 'Are you sure you want to exit?', 'center', maxHeight / 2 - 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press esc to confirm exit.', 'center', maxHeight / 2 + 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press any other button to go to continue.', 'center', maxHeight / 2 + 200, [0 0 0], 100);
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
                Screen('CloseAll');
                ShowCursor;
                ListenChar(0);
                Priority(0);
                return;
            end
        end
    end
    
        
    % check if participant wants to proceed (only for practice)
    DrawFormattedText(windowPtr, 'You have completed the practice sections.', 'center', maxHeight / 2 - 100, [0 0 0], 100);
    DrawFormattedText(windowPtr, 'Pess any button to continue on to the main experiment.', 'center', maxHeight / 2, [0 0 0], 100);
    DrawFormattedText(windowPtr, 'Pess esc to exit the experiment.', 'center', maxHeight / 2 + 100, [0 0 0], 100);
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
        Screen('CloseAll');
        ShowCursor;
        ListenChar(0);
        Priority(0);
        return;
    else
        continueToMain = 1;
    end
    
end

if practice == 2
    % Instructions Part 3
    expStage = "Practice3";
    exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);

    %exit the script if user hit exit
    if exited
        Screen('CloseAll');
        ShowCursor;
        ListenChar(0);
        Priority(0);
        return;
    end
    
    % Practice Part 3
    pracAcc = 0;
    prac3count = 1;
    while pracAcc < pracAccCutoff/100 || prac3count == 1
        [pracAcc, exited] = practiceGlobalLocal(windowPtr,fontDefault, maxWidth, maxHeight, colorMapping, ...
        taskMapping, pracTrialsPerBlock, prac3count, fixInterval, stimInterval, subID, subjectDir, ifi);
        prac3count = prac3count + 1;
        
        %exit the script if user hit exit
        if exited
            Screen('CloseAll');
            ShowCursor;
            ListenChar(0);
            Priority(0);
            return;
        end

        %give feedback based on accuracy
        Screen(windowPtr,'TextSize',36);
        Screen('TextStyle',windowPtr, 0);
        accuracy = round(pracAcc * 100, 2);
        if pracAcc < pracAccCutoff/100
            text = char("Your accuracy for that block was " + accuracy + ...
                "%.\n\nRemember, you need at least " + pracAccCutoff + "% correct. \n\n\n\nPress space to try again.");
            DrawFormattedText(windowPtr, text, 'center', 'center', [0 0 0]);
        else
            text = char("Great job. Your accuracy for that block was " + accuracy + "%.\n\n\n\nPress space to continue.");
            DrawFormattedText(windowPtr, text , 'center', 'center', [0 0 0]);
        end

        %Wait for participant to hit space then continue to next thing,
        %else esc exits that experiment.
        Screen('Flip',windowPtr);
        WaitSecs(0.5);
        spaceHit = 0;
        exited = 0;
        while ~spaceHit && ~exited
            [keyPressed,secs,keyCode] = KbCheck;
            if keyCode(escape) == 1
                exited = 1;
            elseif keyCode(space) == 1
                spaceHit = 1;
            end
        end
                
        if exited
            %are you sure?
            DrawFormattedText(windowPtr, 'Are you sure you want to exit?', 'center', maxHeight / 2 - 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press esc to confirm exit.', 'center', maxHeight / 2 + 100, [0 0 0], 100);
            DrawFormattedText(windowPtr, 'Press any other button to go to continue.', 'center', maxHeight / 2 + 200, [0 0 0], 100);
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
                Screen('CloseAll');
                ShowCursor;
                ListenChar(0);
                Priority(0);
                return;
            end
        end
    end
    
        
    % check if participant wants to proceed (only for practice)
    DrawFormattedText(windowPtr, 'You have completed the practice section.', 'center', maxHeight / 2 - 100, [0 0 0], 100);
    DrawFormattedText(windowPtr, 'Pess any button to continue on to the main experiment.', 'center', maxHeight / 2, [0 0 0], 100);
    DrawFormattedText(windowPtr, 'Pess esc to exit the experiment.', 'center', maxHeight / 2 + 100, [0 0 0], 100);
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
        Screen('CloseAll');
        ShowCursor;
        ListenChar(0);
        Priority(0);
        return;
    else
        continueToMain = 1;
    end
    
end

% % Instructions for Main
% if continueToMain == 1 %if did practice and now continuing to main task
%     expStage = "Main2";
%     exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);
% else
%     expStage = "Main1";
%     exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);
% end

expStage = "Main1";
exited = instructions(windowPtr,expStage, maxWidth, maxHeight, pracAccCutoff, pracTrialsPerBlock, pracOrder, taskMapping, colorMapping);

%exit the script if user hit exit
if exited
    Screen('CloseAll');
    ShowCursor;
    ListenChar(0);
    Priority(0);
    return;
end
    
% Main Task
exited = mainTask(windowPtr, fontDefault, maxWidth, maxHeight, blockOrder, colorMapping, ...
taskMapping, fixInterval, stimInterval, trialsPerBlock, subID, subjectDir, ifi, completedMainBlocks);


%% End of Experiment

%End of experiment screen
Screen(windowPtr,'TextSize',36);
Screen('TextStyle',windowPtr, 0);
DrawFormattedText(windowPtr,'You have reached the end of the experiment. \n\nThank you for your participation.\n\n\n\n Press any button to exit.', 'center', 'center', [0 0 0]);
Screen('Flip',windowPtr);

%way for participant/experimenter to close experiment
keyPressed = 0;
while ~keyPressed
    [keyPressed,secs,keyCode] = KbCheck;
end

%Close experiment
Screen('CloseAll');
ShowCursor;
ListenChar(0);
Priority(0);