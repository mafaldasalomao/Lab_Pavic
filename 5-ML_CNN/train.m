clc; clear;


imds = imageDatastore('database', 'IncludeSubfolders',true, 'LabelSource','foldernames');
[trainGestureData, testGestureData] = splitEachLabel(imds, 0.90, 'randomized');

layers = [
    imageInputLayer([256 256 1])
    convolution2dLayer(5, 20)
    reluLayer
    maxPooling2dLayer(2, 'Stride',2);
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer()    
];

options = trainingOptions('sgdm',...
    'Plots','training-progress',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',5,...
    'MiniBatchSize',300);

options.MaxEpochs = 30;
options.InitialLearnRate = 0.0001;

convnet = trainNetwork(trainGestureData, layers, options);
save("trainedNetwork.net", "convnet");

[labels, err_test] = classify(convnet, testGestureData, 'MiniBatchSize',300);
confusionchart(testGestureData.Labels, labels);
figure;
plotconfusion(testGestureData.Labels, labels);

