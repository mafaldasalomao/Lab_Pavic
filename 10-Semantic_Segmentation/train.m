data = load('triangleSegmentationNetwork.mat');
net = data.net

net.Layers
I = imread('triangleTest.jpg');
figure(1); imshow(I);
[C, scores] = semanticseg(I, net, 'MiniBatchSize', 30);
B = labeloverlay(I, C);
figure(2); imshow(B, []);
figure(3); imagesc(scores)
axis square
colorbar
BW = C == 'triangle';
figure(4); imshow(BW)

