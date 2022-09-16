clc; clear all ; close all;

SceneImage = rgb2gray(imread("scene.jpg"));
TargetImage = rgb2gray(imread("crop.jpg"));
%extractLBPFeatures
ScenePoints = detectSURFFeatures(SceneImage);
TargetPoints = detectSURFFeatures(TargetImage);

[SceneFeatures, SceneValidPoints] = extractFeatures(SceneImage, ScenePoints);
[TargetFeatures, TargetValidPoints] = extractFeatures(TargetImage, TargetPoints);

indexPairs = matchFeatures(SceneFeatures, TargetFeatures);
matchedPoints1 = SceneValidPoints(indexPairs(:,1));
matchedPoints2 = TargetValidPoints(indexPairs(:,2));

figure;

showMatchedFeatures(SceneImage, TargetImage, matchedPoints1, matchedPoints2, 'montage');

