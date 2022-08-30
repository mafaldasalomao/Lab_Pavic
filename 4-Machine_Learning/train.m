trainingDS = imageDatastore('database', 'IncludeSubfolders',true, 'LabelSource','foldernames');
cellSize = [8 8];
numImages = 150;
trainingFeatures = [];
for i=1: numImages
    img = readimage(trainingDS, i);
    trainingFeatures(i, :) = extractHOGFeatures(img, cellSize);
end

trainingLabels = trainingDS.Labels;

classifier = fitcecoc(trainingFeatures,trainingLabels);
save classifier classifier

msgbox('Created Classifier Successfully')

