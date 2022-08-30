load flower
subplot(1,2,1)
imagesc(flower.Orig)
title('Original')
subplot(1,2,2)
imagesc(flower.Noisy)
title('Noisy')
colormap gray

load colorflower
subplot(1,2,1)
imagesc(colorflower.Orig)
title('Original')
subplot(1,2,2)
imagesc(colorflower.Noisy)
title('Noisy')

imden = wdenoise2(colorflower.Noisy,2,'CycleSpinning',1);
figure
subplot(1,2,1)
imagesc(imden)
title('Denoised')
subplot(1,2,2)
imagesc(colorflower.Noisy)
title('Noisy')