load classifier;
cellSize = [8 8];
cd testing;
[filename, pathname] = uigetfile('*.*', 'Pick a matlab code');
filename = strcat(pathname, filename);
img = imread(filename);
figure (1); imshow(img);
testFeatures = extractHOGFeatures(img, 'CellSize',cellSize);
[hog, vis] = extractHOGFeatures(img, 'CellSize',cellSize);
figure(2); plot(vis);

%predict

predictedLabels = predict(classifier, testFeatures);
clc; 
disp('class');
disp(predictedLabels);
