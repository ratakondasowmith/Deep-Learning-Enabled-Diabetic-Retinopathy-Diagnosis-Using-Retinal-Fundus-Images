clc;
clear all;
close all;
close all hidden;
warning off;

%% Step 1: Load the input image
[file, path] = uigetfile('*.*', 'Select an image');
img = imread([path, file]);
img = imresize(img, [224 224]);
figure, imshow(img);
title('Input Image');

%% Step 2: Preprocess the image (Noise Removal using Median Filtering)
redChannel = img(:, :, 1);
greenChannel = img(:, :, 2);
blueChannel = img(:, :, 3);

filteredRed = medfilt2(redChannel);
filteredGreen = medfilt2(greenChannel);
filteredBlue = medfilt2(blueChannel);

filteredRGB = cat(3, filteredRed, filteredGreen, filteredBlue);
figure, imshow(filteredRGB);
title('Noise Removed Image');

%% Step 3: Contrast Enhancement (Restored Image)
restoredImage = imadjust(filteredRGB, [0.2, 0.8], [0, 1]);
figure; 
imshow(restoredImage);  
title('Restored Image');

%% Step 4: Image Processing
Segmented_Image = MFORG(restoredImage);
figure; 
imshow(Segmented_Image);
title('Overlay of Segmented Image on Original Image');

%%

matlabroot = cd;    % Dataset path
datasetpath = fullfile(matlabroot,'Segmented Dataset 1');   %Build full file name from parts
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,'randomized');     %Split ImageDatastore labels by proportions

augimdsTrain = augmentedImageDatastore([224 224 3], imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3], imdsValidation);

%%

net = densenet201

layers = [imageInputLayer([224 224 3])
    
    net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers
    
    fullyConnectedLayer(2) % modifying the fullyconnected layer with respect to classes
    
    softmaxLayer
    
    classificationLayer];


% Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress');


[net, traininfo] = trainNetwork(augimdsTrain,layers,options);  %Train neural network for deep learning
% 
% load net
% load traininfo

YPred = classify(net,Segmented_Image);
msgbox(char(YPred))

Accuracy = mean(traininfo.TrainingAccuracy);
fprintf('The classified output using DenseNet is : %f\n',Accuracy);

%%

output = char(YPred);

if strcmp(output, 'Diabetic Retinopathy')
     Diseasematlabroot = cd;    % Dataset path
     Diseasedatasetpath = fullfile(Diseasematlabroot,'Segmented Dataset 2');   %Build full file name from parts
     Diseaseimds = imageDatastore(Diseasedatasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

     [DiseaseimdsTrain, DiseaseimdsValidation] = splitEachLabel(Diseaseimds,0.8);     %Split ImageDatastore labels by proportions

     DiseaseaugimdsTrain = augmentedImageDatastore([224 224 3],DiseaseimdsTrain);  %Generate batches of augmented image data
     DiseaseaugimdsValidation = augmentedImageDatastore([224 224 3],DiseaseimdsValidation);
        % Training Options
     
     Diseaseoptions = trainingOptions('sgdm', ...
           'MiniBatchSize', 64, ...
           'MaxEpochs', 20, ...
           'InitialLearnRate', 0.001, ...
           'ValidationData', DiseaseaugimdsValidation, ...
           'ValidationFrequency', 10, ...
           'Plots', 'training-progress');


     Diseaselayers = [imageInputLayer([224 224 3])
    
             net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers
    
             fullyConnectedLayer(4) % modifying the fullyconnected layer with respect to classes
    
             softmaxLayer
    
             classificationLayer];
     
%       [Diseasenet, Diseasetraininfo] = trainNetwork(DiseaseaugimdsTrain,Diseaselayers,Diseaseoptions);  %Train neural network for deep learning
 
     load Diseasenet
     load Diseasetraininfo

       [DiseaseYPred,Diseasescore] = classify(Diseasenet,Segmented_Image);      %Classify data using a trained deep learning neural network
        msgbox(char(DiseaseYPred));

       DiseaseAccuracy = mean(Diseasetraininfo.TrainingAccuracy);
       fprintf('The classified Disease output using DenseNet is : %f\n',DiseaseAccuracy);
       
       DiseaseYPred = char(DiseaseYPred);    
end

YPred = char(YPred); % Ensure YPred is a character array

if strcmp(YPred, 'Healthy')
    data1 = 0;  % Numeric value for Healthy
    data2 = NaN; 
elseif strcmp(YPred, 'Diabetic Retinopathy')
    data1 = 1;  % Numeric value for Diabetic Retinopathy
    
    DiseaseYPred = char(DiseaseYPred); 
    
    if strcmp(DiseaseYPred, 'Mild DR')
        data2 = 0;  
    elseif strcmp(DiseaseYPred, 'Moderate DR')
        data2 = 1; 
    elseif strcmp(DiseaseYPred, 'Proliferative DR') 
        data2 = 2;  
    elseif strcmp(DiseaseYPred, 'Severe DR')
        data2 = 3;  
    else
        data2 = NaN; 
    end
end

Channel_ID = 2856158;            % Replace with your channel ID
Write_API_Key = 'UG81W8GGKRSFXGG0'; % Replace with your Write API Key

% Send the data to ThingSpeak
thingSpeakWrite(Channel_ID, [data1, data2], 'Fields', [1, 2], 'WriteKey', Write_API_Key);

disp('Data sent to ThingSpeak successfully!');
%%