clc; clear all; close all;
load iris_dataset.mat;

InputFeatures = irisInputs;
TargetClass=irisTargets;
net=patternnet(10);
view(net);
net=train(net, InputFeatures, TargetClass);
view(net);
save net net