I = imread('eight.tif');
imshow(I)
var_speckle = 0.01;
J = imnoise(I,'speckle',var_speckle);
imshow(J)
MedFilter = medfilt2(J, [5 5]);
Weiner= wiener2(J, [5 5]);

figure(); imshow(MedFilter);

figure(); imshow(Weiner);