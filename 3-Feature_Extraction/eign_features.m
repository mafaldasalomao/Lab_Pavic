I = imread('target.png');
I = rgb2gray(I);
points = detectMinEigenFeatures(I);
figure(1); imshow(I); hold on;
plot(points);