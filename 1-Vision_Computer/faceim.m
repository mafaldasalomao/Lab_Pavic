clc;
clear all; close all;

%I = imread("human.jpg");
camObj = webcam(1);
I = snapshot(camObj);

figure(1); imshow(I);

faceDetector = vision.CascadeObjectDetector();

B = step(faceDetector, I);
C = insertShape(I,"Rectangle",B);
figure(2); imshow(C);

%The cascade object detector uses the Viola-Jones algorithm to detect people’s
% faces, noses, eyes, mouth, or upper body. You can also use the Image Labeler
% to train a custom classifier to use with this System object. For details on
% how the function works, see Get Started with Cascade Object Detector.

%The function can train the model using Haar-like features,
% histograms of oriented gradients (HOG), or local binary patterns (LBP)