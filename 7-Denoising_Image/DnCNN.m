clear all; close all; clc;
net = denoisingNetwork("dncnn");
I = imread('teste.jpeg');
%imshow(I)
J = imnoise(I,'gaussian', 0, 0.005);
imshow(J)
denoisedI = denoiseImage(rgb2gray(I),net);
figure(2); imshow(denoisedI)