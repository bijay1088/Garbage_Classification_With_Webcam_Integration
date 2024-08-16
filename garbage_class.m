clear;
close all;
clc;

%%
% Creating a datastore
imds = imageDatastore('garbage_classification', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%%
%calculating total number of images in dataset 
% (Original dataset had 15,150)
totalImages = numel(imds.Files);
disp(['Total number of images: ', num2str(totalImages)]);

%%
%displaying each classes in chart
labelCount = countEachLabel(imds);

figure;
b = bar(labelCount{:,2});
xlabel('Class');
ylabel('Number of Images');
title('Number of Images per Class');

% Generate a larger set of unique colors
numClasses = height(labelCount);
colors = distinguishable_colors(numClasses);

% Set unique colors for each bar
b.FaceColor = 'flat';
for k = 1:numClasses
    b.CData(k,:) = colors(k,:);
end

%adding label
set(gca, 'XTickLabel', labelCount{:,1});


% generating unique colors
function colors = distinguishable_colors(n_colors)
    colors = hsv(n_colors); 
end
%%
% Dividing the dataset
[imdsTrain, imdsRest] = splitEachLabel(imds, 0.75, 'randomized'); % 75% for training
[imdsValidation, imdsTest] = splitEachLabel(imdsRest, 0.6, 'randomized'); % 15% validation, 10% test

%%
% Displaying some images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16);
figure
for i = 1:16
    subplot(4, 4, i)
    imshow(readimage(imdsTrain, idx(i)))
    title(string(imdsTrain.Labels(idx(i))));
end

%%
% Loading pre-trained network
net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;

%%
% Replacing final layers
layersTransfer = net.Layers(1:end-3); %remove the final 3 layers of pretained network
numClasses = numel(categories(imdsTrain.Labels));
%adding layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer]; 

%%

%data preprocessing for better accuracy
pixelRange = [-10 10];
rotationRange = [-20 20];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange, ...
    'RandRotation', rotationRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb'); 

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%%
%displaying the augmented image next to original

% Display the original and augmented images
figure;
for i = 1:2
    idx = randi(numel(imdsTrain.Files)); %selecting random imaeges
    originalImage = readimage(imdsTrain, idx);
    augmentedImage = augment(imageAugmenter, originalImage);

    % Display the original and augmented images side by side
    subplot(2, 3, (i-1)*3 + 1);
    imshow(originalImage);
    title('Original Image ');

    subplot(2, 3, (i-1)*3 + 2);
    imshow(augmentedImage);
    title('Augmented Image ');
end


%%

%adding some options for training
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 50, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

%%
%training the network
netTransfer = trainNetwork(augimdsTrain, layers, options);

%%
%saving the trained model
save('trainedGarbageClassNet.mat', 'netTransfer', 'options', 'augimdsTrain', 'augimdsValidation', 'augimdsTest');

%%
% Load the trained network and the variables

loadedData = load('trainedGarbageClassNet.mat');
netTransfer = loadedData.netTransfer;
augimdsValidation = loadedData.augimdsValidation;
augimdsTest = loadedData.augimdsTest;

%%
% Classifying test images
[predLabel, scores] = classify(netTransfer, augimdsTest);

%%
%checking accuracy
actualLabel = imdsTest.Labels;
accuracy = mean(predLabel == actualLabel) * 100;
fprintf('\n----------------------------\nTest Accuracy: %.2f%%\n', accuracy);
fprintf('----------------------------\n\n');


%%
% Displaying some test images with predictions
idx = randperm(numel(imdsTest.Files), 4);
figure
for i = 1:4
    subplot(2, 2, i)
    imshow(readimage(imdsTest, idx(i)))
    label = predLabel(idx(i));
    title(['Pred: ' char(label) ', Actual: ' char(actualLabel(idx(i)))]);
end
%%
% Confusion matrix
plotconfusion(actualLabel, predLabel);

%%


%%
%precision, recall and f1
% Calculate confusion matrix
confMat = confusionmat(actualLabel, predLabel);

% Precision, recall, F1-score for each class
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);

classNames = categories(actualLabel);

% Bar chart for precision, recall, F1-score
figure;
bar([precision recall f1Score]);
set(gca, 'XTickLabel', classNames);
xlabel('Class');
ylabel('Score');
title('Precision, Recall, and F1-Score for Each Class');
legend('Precision', 'Recall', 'F1-Score');



%%
%displaying incorrect images
incorrectIdx = find(predLabel ~= actualLabel);
figure;
for i = 1:min(16, numel(incorrectIdx))
    subplot(4, 4, i);
    imshow(readimage(imdsTest, incorrectIdx(i)));
    title(['Pred: ' char(predLabel(incorrectIdx(i))) ', Actual: ' char(actualLabel(incorrectIdx(i)))]);
end
