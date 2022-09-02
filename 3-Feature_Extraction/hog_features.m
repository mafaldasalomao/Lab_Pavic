I = imread('teste.jpeg');
[hog2, ptVis] = extractHOGFeatures(I,'CellSize',[5 5] );
figure;
imshow(I); 
plot(ptVis, 'Color', 'green');