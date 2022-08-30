numObservations = 25;
ZNew = randn(numLatentInputs,numObservations,"single");
ZNew = dlarray(ZNew,"CB");

if canUseGPU
    ZNew = gpuArray(ZNew);
end

XGeneratedNew = predict(netG,ZNew);

I = imtile(extractdata(XGeneratedNew));
I = rescale(I);
figure
image(I)
axis off
title("Generated Images")