I = imread('eight.tif');
imshow(I)
J = imnoise(I,'salt & pepper',0.02);
imshow(J)
MedFilter = medfilt2(J, [5 5]);
Weiner= wiener2(J, [5 5]);

figure(); imshow(MedFilter);

figure(); imshow(Weiner);