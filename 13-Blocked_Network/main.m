

%u-net bclocked
unetBlock = @(block) [
    convolution2dLayer(3,2^(5+block))
    reluLayer
    convolution2dLayer(3,2^(5+block))
    reluLayer
    maxPooling2dLayer(2,"Stride",2)];

net = blockedNetwork(unetBlock,4,"NamePrefix","encoder_")
net = initialize(net,dlarray(zeros(224,224,3),"SSC"));
analyzeNetwork(net)

%%Load net pre trained

