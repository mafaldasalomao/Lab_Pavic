function [matchedPoints1,matchedPoints2] = featurextract(SceneImage,TargetImage)
%FEATUREXTRACT Summary of this function goes here
%   Detailed explanation goes here
ScenePoints = detectMinEigenFeatures(SceneImage);
TargetPoints = detectMinEigenFeatures(TargetImage);

[SceneFeatures, SceneValidPoints] = extractFeatures(SceneImage, ScenePoints);
[TargetFeatures, TargetValidPoints] = extractFeatures(TargetImage, TargetPoints);

indexPairs = matchFeatures(SceneFeatures, TargetFeatures);
matchedPoints1 = SceneValidPoints(indexPairs(:,1));
matchedPoints2 = TargetValidPoints(indexPairs(:,2));

end

