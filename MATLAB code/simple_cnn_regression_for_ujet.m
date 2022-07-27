clear all; close all;

my_dir = pwd;
backslashes = strfind(my_dir,filesep);
data_dir = my_dir(1:backslashes(end)-1) + "\MATLAB data\Train Data";

for i = 1:100
    sample = load(data_dir + "\sample" + i + ".dat");
    data(:,:,1,i) = sample(1:44,1:28);
end

indexes = randperm(size(data, 4));

X = data(:,:,:,indexes);
Y = (indexes / 100) - 0.01;
Y = transpose(Y);

X_train = X(:,:,:,1:60);
Y_train = Y(1:60);

X_val = X(:,:,:,61:90);
Y_val = Y(61:90);

X_test = X(:,:,:,91:100);
Y_test = Y(91:100);

layers = [
    imageInputLayer([44 28 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];

miniBatchSize  = 8;
validationFrequency = floor(numel(Y_train)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{X_val,Y_val}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(X_train,Y_train,layers,options);

Y_pred = predict(net,X_test);
%prediction_error = Y_test - Y_pred;

%squares = prediction_error.^2;
%rmse = sqrt(mean(squares))

%me = mean(abs(prediction_error))

%compare_test_pred = [Y_test, Y_pred];


%run('C:\Users\eding\Documents\402\ENS492\MATLAB codes\deep-google\dg_setup');
%res = invert_nn(net, 0.5, [options]);
res = invert_nn(net, Y_pred(1));
res.output{end}

