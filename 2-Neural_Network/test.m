clc; clear all; close all;
load iris_dataset.mat;
load net;
InputValues=input('enter your feature');
feature = irisInputs(:, InputValues);
class=net(feature);

class=vec2ind(class);
disp(class);