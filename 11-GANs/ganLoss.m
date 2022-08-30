function [lossG,lossD] = ganLoss(YReal,YGenerated)

% Calculate the loss for the discriminator network.
lossD = -mean(log(YReal)) - mean(log(1-YGenerated));

% Calculate the loss for the generator network.
lossG = -mean(log(YGenerated));

end