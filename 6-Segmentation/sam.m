clc;
clear;
I = imread('technics.png');
[r c p] =size(I);

R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
mask = zeros(r, c);
maskzeros = zeros(r, c);
threshold = 100;
Mask = (B>threshold);