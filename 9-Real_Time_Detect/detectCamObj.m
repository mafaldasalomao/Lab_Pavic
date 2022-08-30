clc; clear all; close all;
camObj = webcam(1);
TargetImage = imread("target.png");
TargetImage = rgb2gray(TargetImage);

while(1)
    SceneImage1 = snapshot(camObj);
    SceneImage = rgb2gray(SceneImage1);
    [matchedScenePoints, matchedBoxPoints] = featurextract(SceneImage,TargetImage);
    len = length(matchedScenePoints);
    if len>20
        disp('obj detected');
        bbox = [1, 1; ...
            size(TargetImage, 2), 1;...
            size(TargetImage, 2), size(TargetImage, 1);...
            1, size(TargetImage, 1);...
            1, 1];

        [tform, inlierBoxPoints, inlierScenePoints] = ...
            estimateGeometricTransform(matchedBoxPoints,matchedScenePoints, 'affine');
        newBoxPolign = transformPointsForward(tform, bbox)
        d='object recognized'
        currFrame = SceneImage1;
        position = [321 430];
        box_color = {'yellow'};
        RGB = insertText(currFrame, position, d, 'FontSize',22, 'BoxColor',box_color);
        imshow(RGB);
        hold on;
        line(newBoxPolign(:,1), newBoxPolign(:,2), 'Color', 'y' );
        title('detected box');
    else
        disp('object not recognized');
        d='Object not recognized'
        currentFrame = SceneImage1;
        position = [321 430];
        box_color = {'yellow'};
        RGB = insertText(currentFrame, position, d, 'FontSize',22, 'BoxColor',box_color);
        imshow(RGB);
        hold on;
    end
end