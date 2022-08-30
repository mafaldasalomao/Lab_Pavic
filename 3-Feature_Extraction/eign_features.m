I = imread('cameraman.tif');
points = detectMinEigenFeatures(I);
figure(1); imshow(I); hold on;
plot(points);