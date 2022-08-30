I = imread('cameraman.tif');
[hog2, ptVis] = extractHOGFeatures(I,'CellSize',[10 10] );
figure;
imshow(I); 
plot(ptVis, 'Color', 'green');