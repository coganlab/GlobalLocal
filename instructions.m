function exited = instructions(windowPtr, expStage, maxWidth, maxHeight, accCutoff, pracTrialsPerBlock, pracOrder, taskMapping, colorMapping)

    % Ensure only space bar proceeds to next instruction
    space = KbName('space');
    escape = KbName('ESCAPE');
    exited = 0;
    
    % Set looping variables
    displaying_instructions = 1;
    page = 1;
    
    %counterbalancing variables
    if taskMapping == 1
        finger_S = 'right hand index finger';
        key_S = 'left arrow';
        finger_H = 'right hand middle finger';
        key_H = 'right arrow';
    else
        finger_S = 'right hand middle finger';
        key_S = 'right arrow';
        finger_H = 'right hand index finger';
        key_H = 'left arrow';
    end
    
    if colorMapping == 1
        globalColor = 'red';
        localColor = 'blue';
    else
        globalColor = 'blue';
        localColor = 'red';
    end
    

    %instructions
    while displaying_instructions
        
        %implement defaults at every iteration
        Screen('TextStyle',windowPtr, 0);
        Screen(windowPtr,'TextSize',24);
        Screen(windowPtr,'TextFont','Arial');
        
        switch expStage
            case "Opener"
                switch page
                    case 1
                        %instruction text
                        instruction = char("In this task, you will see one of the following on each trial.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 - 300, [0 0 0], 70);
                        instructions_img = Screen('MakeTexture', windowPtr, imread('instruction_images/task_stimuli.png')); %dims are 900 x 278 pixels, 450 278/2 
                        Screen('DrawTexture', windowPtr, instructions_img, [], [maxWidth/2 - 360, maxHeight/2 - 222/2 - 150, maxWidth/2 + 360, maxHeight/2 + 222/2 - 150,]); 
                        instruction = char("You will indicate either the small letters or the large shape, based on the color of the rectangle around the letters.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 + 10, [0 0 0], 70);
                        
                        %task images
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(1))); %dims are 500 x 246 pixels 
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200 - 220, maxHeight/2 + 80, maxWidth/2 + 200 - 220, maxHeight/2 + 80 + 196.8]);
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(2))); %dims are 500 x 246 pixels 
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200 + 220, maxHeight/2 + 80, maxWidth/2 + 200 + 220, maxHeight/2 + 80 + 196.8]);
                        
                        %press space to continue
                        Screen('TextStyle',windowPtr, 1);
                        instruction = char("Press space to continue.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 + 325, [0 0 0], 70);
                        
                        %display text and wait for space press
                        Screen('Flip',windowPtr);
                        WaitSecs(0.5);
                        keyPressed = 0;
                        while ~keyPressed
                            [keyPressed,secs,keyCode] = KbCheck;
                            if keyCode(space) == 1
                                %end displaying instructions, proceed to task
                                displaying_instructions = 0;
                            elseif keyCode(escape) == 1
                                exited = 1;
                            end
                        end
                        
                        if exited
                            %are you sure you want to exit?
                            drawExitWarning();
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
                end %switch page Opener
            
            %Instructions Preceeding Practice 1
            case "Practice1"
                switch page
                    case 1
                        %instruction text
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(1))); %dims are 500 x 246 pixels
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200, maxHeight/2 - 300, maxWidth/2 + 200, maxHeight/2 - 300 + 196.8]);
                        
                        instruction = char("In this practice task, indicate if the " + first_task_target + " is an 'S' or an 'H', using the left and right arrow keys.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 - 25, [0 0 0], 70);

                        instructions_img = Screen('MakeTexture', windowPtr, imread('instruction_images/arrow_keys.png')); %600 x 216 40% reduced
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 - 200, maxHeight/2 + 70, maxWidth/2 + 180 - 200, maxHeight/2 + 70 + 129.6]);
                        
                        instructions_img = Screen('MakeTexture', windowPtr, imread(['instruction_images/taskmapping', num2str(taskMapping), '.png'])); %600 x 238
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 + 200, maxHeight/2 + 70, maxWidth/2 + 180 + 200, maxHeight/2 + 70 + 142.8]);
                        
                        Screen('TextStyle',windowPtr, 1);
                        instruction = char("Press space to begin.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 + 300, [0 0 0], 70);
                        
                        %display text and wait for space press
                        Screen('Flip',windowPtr);
                        WaitSecs(0.5);
                        keyPressed = 0;
                        while ~keyPressed
                            [keyPressed,secs,keyCode] = KbCheck;
                            if keyCode(space) == 1
                                %end displaying instructions, proceed to task
                                displaying_instructions = 0;
                            elseif keyCode(escape) == 1
                                exited = 1;
                            end
                        end
                        
                        if exited
                            %are you sure you want to exit?
                            drawExitWarning();
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
                end %case practice 1
                
             %Instructions Preceeding Practice 1
            case  "Practice2"
                switch page
                    case 1
                        %instruction text
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(2))); %dims are 500 x 246 pixels
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200, maxHeight/2 - 300, maxWidth/2 + 200, maxHeight/2 - 300 + 196.8]);
                        
                        instruction = char("In this practice task, indicate if the " + second_task_target + " is an 'S' or an 'H', using the left and right arrow keys.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 - 25, [0 0 0], 70);
          
                        instructions_img = Screen('MakeTexture', windowPtr, imread('instruction_images/arrow_keys.png')); %600 x 216 40% reduced
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 - 200, maxHeight/2 + 70, maxWidth/2 + 180 - 200, maxHeight/2 + 70 + 129.6]);
                        
                        instructions_img = Screen('MakeTexture', windowPtr, imread(['instruction_images/taskmapping', num2str(taskMapping), '.png'])); %600 x 238
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 + 200, maxHeight/2 + 70, maxWidth/2 + 180 + 200, maxHeight/2 + 70 + 142.8]);
                        
                        Screen('TextStyle',windowPtr, 1);
                        instruction = char("Press space to begin.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 + 300, [0 0 0], 70);
                        
                        %display text and wait for space press
                        Screen('Flip',windowPtr);
                        WaitSecs(0.5);
                        keyPressed = 0;
                        while ~keyPressed
                            [keyPressed,secs,keyCode] = KbCheck;
                            if keyCode(space) == 1
                                displaying_instructions = 0;
                            elseif keyCode(escape) == 1
                                exited = 1;
                            end
                        end
                        
                        if exited
                            %are you sure you want to exit?
                            drawExitWarning();
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
                end %case practice 2
                
            %Instructions Preceeding Practice 3
            case  "Practice3"
                switch page
                    case 1
                        %instruction text
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(1))); %dims are 500 x 246 pixels 
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200 - 220, maxHeight/2 - 300, maxWidth/2 + 200 - 220, maxHeight/2 - 300 + 196.8]);
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(2))); %dims are 500 x 246 pixels 
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200 + 220, maxHeight/2 - 300, maxWidth/2 + 200 + 220, maxHeight/2 - 300 + 196.8]);
                        
                        instruction = char("In this practice task, indicate either the large shape or the small letters, based on the color of the rectangle.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 - 25, [0 0 0], 70);
                                         
                        instructions_img = Screen('MakeTexture', windowPtr, imread('instruction_images/arrow_keys.png')); %600 x 216 40% reduced
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 - 200, maxHeight/2 + 70, maxWidth/2 + 180 - 200, maxHeight/2 + 70 + 129.6]);
                        
                        instructions_img = Screen('MakeTexture', windowPtr, imread(['instruction_images/taskmapping', num2str(taskMapping), '.png'])); %600 x 238
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 + 200, maxHeight/2 + 70, maxWidth/2 + 180 + 200, maxHeight/2 + 70 + 142.8]);
                        
                        Screen('TextStyle',windowPtr, 1);
                        instruction = char("Press space to begin.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 + 300, [0 0 0], 70);
                        
                        %display text and wait for space press
                        Screen('Flip',windowPtr);
                        WaitSecs(0.5);
                        keyPressed = 0;
                        while ~keyPressed
                            [keyPressed,secs,keyCode] = KbCheck;
                            if keyCode(space) == 1
                                displaying_instructions = 0;
                            elseif keyCode(escape) == 1
                                exited = 1;
                            end
                        end
                        
                        if exited
                            %are you sure you want to exit?
                            drawExitWarning();
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
                end %case practice 3
                
            %Instructions Preceeding Main Task
            case  "Main1"
                switch page
                    case 1
                        %instruction text
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(1))); %dims are 500 x 246 pixels 
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200 - 220, maxHeight/2 - 300, maxWidth/2 + 200 - 220, maxHeight/2 - 300 + 196.8]);
                        instructions_img = Screen('MakeTexture', windowPtr, imread(get_task_image(2))); %dims are 500 x 246 pixels 
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 200 + 220, maxHeight/2 - 300, maxWidth/2 + 200 + 220, maxHeight/2 - 300 + 196.8]);
                        
                        instruction = char("In this task, indicate either the large shape or the small letters, based on the color of the rectangle.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 - 25, [0 0 0], 70);
                                         
                        instructions_img = Screen('MakeTexture', windowPtr, imread('instruction_images/arrow_keys.png')); %600 x 216 40% reduced
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 - 200, maxHeight/2 + 70, maxWidth/2 + 180 - 200, maxHeight/2 + 70 + 129.6]);
                        
                        instructions_img = Screen('MakeTexture', windowPtr, imread(['instruction_images/taskmapping', num2str(taskMapping), '.png'])); %600 x 238
                        Screen('DrawTexture', windowPtr, instructions_img, [], ...
                            [maxWidth/2 - 180 + 200, maxHeight/2 + 70, maxWidth/2 + 180 + 200, maxHeight/2 + 70 + 142.8]);
                        
                        Screen('TextStyle',windowPtr, 1);
                        instruction = char("Press space to begin.");
                        DrawFormattedText(windowPtr, instruction, 'center', maxHeight / 2 + 300, [0 0 0], 70);
                        
                        %display text and wait for space press
                        Screen('Flip',windowPtr);
                        WaitSecs(0.5);
                        keyPressed = 0;
                        while ~keyPressed
                            [keyPressed,secs,keyCode] = KbCheck;
                            if keyCode(space) == 1
                                %end displaying instructions, proceed to task
                                displaying_instructions = 0;
                            elseif keyCode(escape) == 1
                                exited = 1;
                            end
                        end
                        
                        if exited
                            %are you sure you want to exit?
                            drawExitWarning();
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
                end %case main1
        end 
    end
    
    function target = first_task_target()
        if pracOrder == 1
            target = 'large shape of the letters is';
        else
            target = 'small letters are';
        end
    end

    function target = second_task_target()
        if pracOrder == 1
            target = 'small letters are';
        else
            target = 'large shape of the letters is';
        end
    end

    function color = colorFirstTask()
        if pracOrder == 1
            color = globalColor;
        else
            color = localColor;
        end
    end

    function color = colorSecondTask()
        if pracOrder == 1
            color = localColor;
        else
            color = globalColor;
        end
    end
    
    function task_image = get_task_image(num)
        if num == 1
            if pracOrder == 1
                if colorMapping == 1
                    task_image = 'instruction_images/task_instruction_3.jpg';
                else
                    task_image = 'instruction_images/task_instruction_1.jpg';
                end
            else
                if colorMapping == 1
                    task_image = 'instruction_images/task_instruction_4.jpg';
                else
                    task_image = 'instruction_images/task_instruction_2.jpg';
                end
            end
        else
            if pracOrder == 1
                if colorMapping == 1
                    task_image = 'instruction_images/task_instruction_4.jpg';
                else
                    task_image = 'instruction_images/task_instruction_2.jpg';
                end
            else
                if colorMapping == 1
                    task_image = 'instruction_images/task_instruction_3.jpg';
                else
                    task_image = 'instruction_images/task_instruction_1.jpg';
                end
            end
        end
    end

    function drawExitWarning()
        Screen('TextStyle',windowPtr, 0);
        DrawFormattedText(windowPtr, 'Are you sure you want to exit?', 'center', maxHeight / 2 - 150, [0 0 0], 100);
        DrawFormattedText(windowPtr, 'Press esc to confirm exit.', 'center', maxHeight / 2 + 0, [0 0 0], 100);
        DrawFormattedText(windowPtr, 'Press any other button to go back to the instructions.', 'center', maxHeight / 2 + 150, [0 0 0], 100);
    end
end



