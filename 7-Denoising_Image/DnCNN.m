clear all; close all; clc;
net = denoisingNetwork("dncnn");
I = imread('eight.tif');
%imshow(I)
J = imnoise(I,'gaussian', 0, 0.005);
imshow(J)
denoisedI = denoiseImage(J,net);
figure(2); imshow(denoisedI)