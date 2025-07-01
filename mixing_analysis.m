clear; clc;

% Let user select video file using file picker
[videoName, videoPath] = uigetfile({'*.mp4;*.avi','Video Files (*.mp4, *.avi)'}, 'Select a video file');
if isequal(videoName, 0)
    error('No video file selected.');
end
videoFile = fullfile(videoPath, videoName);

% Prompt for other parameters
prompt = {'Number of Training Frames:', 'Selected Rotor Type:', 'Selected Speed:', 'Output Video File Name:'};
title1 = 'Input Parameters';
dims = [1 50];
defaultInput = {'50', '2', '200', 'output_video.avi'};
inputParams = inputdlg(prompt, title1, dims, defaultInput);

% Parse input
numTrainingFrames = str2double(inputParams{1});
selectedRotorType = str2double(inputParams{2});
selectedSpeed = str2double(inputParams{3});
outputVideoFile = inputParams{4};

% Load the video
vidObj = VideoReader(videoFile);
numFrames = vidObj.NumFrames;

% Initialize arrays
mixingEfficiency = zeros(numFrames, 1);
mixedAreaOverTime = zeros(numFrames, 1);
colorChangeArea = zeros(numFrames, 1);
initialColor = [];

% Setup figure and video writer
figure;
outputVideo = VideoWriter(outputVideoFile);
open(outputVideo);

% Display selected rotor and speed
fprintf('Selected Rotor Type: %d | Speed: %d\n', selectedRotorType, selectedSpeed);

% Estimate initial background
vidObj.CurrentTime = 0;
frameIdx = 1;
backgroundEstimator = vision.ForegroundDetector('NumTrainingFrames', numTrainingFrames, ...
    'InitialVariance', numTrainingFrames*numTrainingFrames);
while hasFrame(vidObj) && frameIdx <= numTrainingFrames
    frame = readFrame(vidObj);
    step(backgroundEstimator, rgb2gray(frame));
    frameIdx = frameIdx + 1;
end

% Process video frames
vidObj.CurrentTime = 0;
frameIdx = 1;
while hasFrame(vidObj)
    frame = readFrame(vidObj);
    grayFrame = rgb2gray(frame);

    % Foreground mask and processing
    foregroundMask = step(backgroundEstimator, grayFrame);
    se = strel('disk', 5);
    enhancedMask = imopen(foregroundMask, se);

    % Mixing metrics
    totalPixels = numel(frame);
    mixingEfficiency(frameIdx) = 1 - (sum(enhancedMask(:)) / totalPixels);
    mixedAreaOverTime(frameIdx) = sum(enhancedMask(:));

    % Color overlay
    turquoiseColor = [64, 224, 208];
    turquoiseMask = cat(3, uint8(enhancedMask) * turquoiseColor(1), ...
                             uint8(enhancedMask) * turquoiseColor(2), ...
                             uint8(enhancedMask) * turquoiseColor(3));
    mixedAreaOverlay = frame + turquoiseMask;

    % Color change detection
    if isempty(initialColor)
        initialColor = double(frame);
    end
    colorChangeArea(frameIdx) = sum(abs(double(frame) - initialColor), 'all');

    % Show raw vs processed
    subplot(1,2,1); imshow(frame); title('Raw Video');
    subplot(1,2,2); imshow(mixedAreaOverlay); title('Region of Mixing Propagation (Turquoise)');
    drawnow;

    % Write to video
    writeVideo(outputVideo, mixedAreaOverlay);
    frameIdx = frameIdx + 1;
end

close(outputVideo);

% Plot results
timeSteps = 1:numFrames;
figure;
subplot(2,1,1);
plot(timeSteps, mixedAreaOverTime);
title('Region of Mixing Propagation');
xlabel('Time Step'); ylabel('Mixed Area');

subplot(2,1,2);
plot(timeSteps, colorChangeArea);
title(sprintf('Detected Mixing Area - Rotor Type: %d | Speed: %d', selectedRotorType, selectedSpeed));
xlabel('Time Step'); ylabel('Mixing Area');
