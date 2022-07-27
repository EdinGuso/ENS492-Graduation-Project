clear all; close all;

%DATA%
%load data
my_dir = pwd;
backslashes = strfind(my_dir,filesep);
data_dir = my_dir(1:backslashes(end)-1) + "\MATLAB data\Train Data";

data = cell(1,100);

for i = 1:100
    sample = load(data_dir + "\sample" + i + ".dat");
    data(1,i) = num2cell(sample(1:44,1:28), [1 2]);
end

%shuffle and divide
indexes = randperm(size(data, 2));

X = data(:,indexes);
Y = (indexes / 100) - 0.01;
Y = transpose(Y);

X_train = X(:,1:60);
Y_train = Y(1:60);

X_val = X(:,61:90);
Y_val = Y(61:90);

X_test = X(:,91:100);
Y_test = Y(91:100);

%AUTOENCODER%
%Training
hiddenSize = 256;
autoenc = trainAutoencoder(X_train,hiddenSize,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.15,...
        'MaxEpochs', 256);

%Testing
X_reconstructed = predict(autoenc,X_test);

%Comparisson of real images and reconstructed images
figure('Name','Real Test Data');
for i = 1:10
    avg_stress = sum(sum(X_test{i})) / (28 * 44);
    min_stress = min(min(X_test{i}));
    max_stress = max(max(X_test{i}));
    subplot(2,5,i);
    imshow(X_test{i});
    title({['Ujet = ', num2str(Y_test(i))]
           ['avg stress = ', num2str(avg_stress)]
           ['min stress = ', num2str(min_stress)]
           ['max stress = ', num2str(max_stress)]});
end
figure('Name','Reconstructed Test Data');
for i = 1:10
    subplot(2,5,i);
    imshow(X_reconstructed{i});
    title(['Ujet = ', num2str(Y_test(i))])
end

%Error calculation
err = 0;
for i = 1:10
    err = err + sum(sum(abs(X_reconstructed{i} - X_test{i})));
end
merr = err / 10;
mmerr = merr / (44*28);

%NEURAL NETWORK%
%Encode the data
X_train_encoded = num2cell(encode(autoenc, X_train), 1);
X_val_encoded = num2cell(encode(autoenc, X_val), 1);
X_test_encoded = num2cell(encode(autoenc, X_test), 1);
Y_train_encoded = num2cell(Y_train)';
Y_val_encoded = num2cell(Y_val)';
Y_test_encoded = num2cell(Y_test)';

%Define the NN architecture
layers = [
    sequenceInputLayer(256,'Name','Encoded Data Input Layer')
    fullyConnectedLayer(64,'Name','Fully Connected Regression Layer-1')
    tanhLayer('Name','Activation Function Layer-1')
    fullyConnectedLayer(16,'Name','Fully Connected Regression Layer-2')
    tanhLayer('Name','Activation Function Layer-2')
    fullyConnectedLayer(1,'Name','Fully Connected Regression Layer-3')
    tanhLayer('Name','Activation Function Layer-3')
    regressionLayer('Name','Output Ujet Layer')];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',1, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{X_val_encoded,Y_val_encoded}, ...
    'ValidationFrequency',1, ...
    'Plots','training-progress', ...
    'Verbose',false);

%Train and test
net = trainNetwork(X_train_encoded,Y_train_encoded,layers,options);
Y_pred = predict(net,X_test_encoded);

%decode inner network manually
decoded_Y_pred = Y_pred;

for i = size(net.Layers)-2:-2:2
    bias = net.Layers(i).Bias;
    weights = net.Layers(i).Weights;
    for j = 1:10
        decoded_Y_pred{j} = pinv(weights) * (atanh(decoded_Y_pred{j}) - bias);
    end 
end

%decode outer network automatically
double_decoded_Y_pred = decode(autoenc, cell2mat(decoded_Y_pred'));

%Comparisson of real images and reconstructed images
figure('Name','Real Test Data');
for i = 1:10
    subplot(2,5,i);
    imshow(X_test{i});
    title(['Ujet = ', num2str(Y_test(i))])
end
figure('Name','Reconstructed Test Data');
for i = 1:10
    subplot(2,5,i);
    imshow(double_decoded_Y_pred{i});
    title(['Ujet = ', num2str(Y_pred{i})])
end

%Plot a side by side comparisson using surf
figure;
subplot(1,2,1); surf(1:44,1:28,X_test{1}'); title('Real Test Data Sample')
subplot(1,2,2); surf(1:44,1:28,double_decoded_Y_pred{1}'); title('Reconstructed Test Data Sample')
