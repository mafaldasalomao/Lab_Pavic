%reduce data, describe original data
%Extract unique core e unique infor
%Reduce missclassification
%Reduce complexity math
im = imread('teste.jpeg');

[LL LH HL HH] = dwt2(im, 'db1'); %haar db

aa = [LL LH; HL HH];

imshow(aa, []);

title('Discrete Wavelet Transform Image')