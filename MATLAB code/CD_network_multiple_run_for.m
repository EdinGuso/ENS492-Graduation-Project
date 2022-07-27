clear all; close all;

bookKeeping = zeros(12,101);


%% PARAMETER INITIALIZATIONS
num_samples = 100;
train_size = 65;
val_size = 25;
test_size = 10; %has to be an even number (for plotting)
sample_x_size = 44;
sample_y_size = 28;

%% LOAD DATA (X)
my_dir = pwd;
backslashes = strfind(my_dir,filesep);
data_dir = my_dir(1:backslashes(end)-1) + "\MATLAB data\Train Data";
data = cell(1,num_samples);
for i = 1:num_samples
    sample = load(data_dir + "\sample" + i + ".dat");
    data(1,i) = num2cell(sample(1:sample_x_size,1:sample_y_size), [1 2]);
    bookKeeping(1,i) = 0.01*(i-1);
end

%% COMPUTE TARGET CD (Y)
CD = [];
for i = 1:num_samples
    oneDintegrated = trapz(data{i}) / sample_x_size;
    CD(i) = trapz(oneDintegrated) / sample_y_size;
    bookKeeping(12,i) = CD(i);
end


for i = 1:1
    close all;
    
    %% SHUFFLE AND SPLIT
    indexes = randperm(size(data, 2));
    
    X = data(:,indexes);
    Y = CD(indexes);
    ujet = (indexes / 100) - 0.01;
    
    X_train = X(:,1:train_size);
    Y_train = Y(1:train_size);
    ujet_train = ujet(1:train_size);
    
    X_val = X(:,train_size+1:train_size+val_size);
    Y_val = Y(train_size+1:train_size+val_size);
    ujet_val = ujet(train_size+1:train_size+val_size);
    
    X_test = X(:,train_size+val_size+1:num_samples);
    Y_test = Y(train_size+val_size+1:num_samples);
    ujet_test = ujet(train_size+val_size+1:num_samples);
    
    
    %% AUTOENCODER LAYER 1
    %Training
    hiddenSize_1 = 256;
    autoenc_1 = trainAutoencoder(X_train,hiddenSize_1,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',2,...
        'SparsityProportion',0.25,...
        'MaxEpochs', 256,...
        'ShowProgressWindow',false);
    
    %Testing
    X_reconstructed = predict(autoenc_1,X_test);
    %
    %     %Comparisson of real images and reconstructed images
    %     figure('Name','Real Test Data');
    %     for i = 1:test_size
    %         subplot(2,test_size/2,i);
    %         imshow(X_test{i});
    %         title(['CD = ', num2str(Y_test(i))])
    %     end
    %     figure('Name','Reconstructed Test Data');
    %     for i = 1:test_size
    %         subplot(2,test_size/2,i);
    %         imshow(X_reconstructed{i});
    %         title(['CD = ', num2str(Y_test(i))])
    %     end
    
    %Error calculation
    err = 0;
    max_val = intmin;
    for j = 1:test_size
        err = err + sum(sum(abs(X_reconstructed{j} - X_test{j})));
        if (max_val < max(max(X_test{j})))
            max_val = max(max(X_test{j}));
        end
    end
    merr = err / test_size;
    mmerr = merr / (sample_x_size*sample_y_size);
    
    autoencoder_layer_1_relative_mean_error = 100 * mmerr / max_val;
    
    %% AUTOENCODER LAYER 2
    %Encode the data
    X_train_encoded = num2cell(encode(autoenc_1, X_train), 1);
    X_val_encoded = num2cell(encode(autoenc_1, X_val), 1);
    X_test_encoded = num2cell(encode(autoenc_1, X_test), 1);
    
    %Training
    hiddenSize_2 = 64;
    autoenc_2 = trainAutoencoder(X_train_encoded,hiddenSize_2,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',2,...
        'SparsityProportion',0.25,...
        'MaxEpochs', 256,...
        'ShowProgressWindow',false);
    
    %Testing
    X_encoded_reconstructed = predict(autoenc_2,X_test_encoded);
    X_reconstructed_reconstructed = decode(autoenc_1,cell2mat(X_encoded_reconstructed));
    
    %     %Comparisson of real images and reconstructed images
    %     figure('Name','Real Test Data');
    %     for i = 1:test_size
    %         subplot(2,test_size/2,i);
    %         imshow(X_test{i});
    %         title(['CD = ', num2str(Y_test(i))])
    %     end
    %     figure('Name','Reconstructed Test Data');
    %     for i = 1:test_size
    %         subplot(2,test_size/2,i);
    %         imshow(X_reconstructed_reconstructed{i});
    %         title(['CD = ', num2str(Y_test(i))])
    %     end
    
    %Error calculation
    err = 0;
    max_val = intmin;
    for k = 1:test_size
        err = err + sum(sum(abs(X_reconstructed_reconstructed{k} - X_test{k})));
        if (max_val < max(max(X_test{k})))
            max_val = max(max(X_test{k}));
        end
    end
    merr = err / test_size;
    mmerr = merr / (sample_x_size*sample_y_size);
    
    autoencoder_layer_2_relative_mean_error = 100 * mmerr / max_val;
    
    %% AUTOENCODER LAYER 3
    %Encode the data
    X_train_encoded_encoded = num2cell(encode(autoenc_2, X_train_encoded), 1);
    X_val_encoded_encoded = num2cell(encode(autoenc_2, X_val_encoded), 1);
    X_test_encoded_encoded = num2cell(encode(autoenc_2, X_test_encoded), 1);
    
    %Training
    hiddenSize_3 = 16;
    autoenc_3 = trainAutoencoder(X_train_encoded_encoded,hiddenSize_3,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',2,...
        'SparsityProportion',0.25,...
        'MaxEpochs', 256,...
        'ShowProgressWindow',false);
    
    %Testing
    X_encoded_encoded_reconstructed = predict(autoenc_3,X_test_encoded_encoded);
    X_reconstructed_reconstructed_reconstructed = decode(autoenc_1,cell2mat(decode(autoenc_2,cell2mat(X_encoded_encoded_reconstructed))));
    
    %     %Comparisson of real images and reconstructed images
    %     figure('Name','Real Test Data');
    %     for i = 1:test_size
    %         subplot(2,test_size/2,i);
    %         imshow(X_test{i});
    %         title(['CD = ', num2str(Y_test(i))])
    %     end
    %     figure('Name','Reconstructed Test Data');
    %     for i = 1:test_size
    %         subplot(2,test_size/2,i);
    %         imshow(X_reconstructed_reconstructed_reconstructed{i});
    %         title(['CD = ', num2str(Y_test(i))])
    %     end
    
    %Error calculation
    err = 0;
    max_val = intmin;
    for m = 1:10
        err = err + sum(sum(abs(X_reconstructed_reconstructed_reconstructed{m} - X_test{m})));
        if (max_val < max(max(X_test{m})))
            max_val = max(max(X_test{m}));
        end
    end
    merr = err / test_size;
    mmerr = merr / (sample_x_size*sample_y_size);
    
    autoencoder_layer_3_relative_mean_error = 100 * mmerr / max_val;
    
    %% SHALLOW NETWORK
    %Encode the data
    X_train_encoded_encoded_encoded = num2cell(encode(autoenc_3, X_train_encoded_encoded), 1);
    X_val_encoded_encoded_encoded = num2cell(encode(autoenc_3, X_val_encoded_encoded), 1);
    X_test_encoded_encoded_encoded = num2cell(encode(autoenc_3, X_test_encoded_encoded), 1);
    
    %Define the NN architecture
    layers = [
        sequenceInputLayer(hiddenSize_3,'Name','Encoded Data Input Layer')
        fullyConnectedLayer(32,'Name','Fully Connected Regression Layer-1')
        tanhLayer('Name','Activation Function Layer-1')
        fullyConnectedLayer(4,'Name','Fully Connected Regression Layer-2')
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
        'ValidationData',{X_val_encoded_encoded_encoded,num2cell(Y_val)}, ...
        'ValidationFrequency',1, ...
        'Plots','none', ...
        'Verbose',false);
    
    %Train and test
    net = trainNetwork(X_train_encoded_encoded_encoded,num2cell(Y_train),layers,options);
    Y_pred = predict(net,X_test_encoded_encoded_encoded);
    
    comp=[Y_test', cell2mat(Y_pred)];
    
    %Error calculation
    err = 0;
    max_val = intmin;
    for n = 1:test_size
        err = err + sum(abs(Y_pred{n} - Y_test(n)));
        col = findPlace(ujet_test(n),bookKeeping);
        bookKeeping(i,col) = Y_pred{n};
        if (max_val < max(Y_test(n)))
            max_val = max(Y_test(n));
        end
    end
    merr = err / test_size;
    
    CD_prediction_relative_mean_error = 100 * merr / max_val;
    
    %     %% RESULTS
    %     %Plot the CD test and predictions
    %     %figure; hold on
    %     %plot1 = plot(Y_test); legend1 = "Y Test";
    %     %plot2 = plot(cell2mat(Y_pred)'); legend2 = "Y Pred";
    %     %legend([plot1,plot2], [legend1, legend2]);
    %
    %     figure;
    %     plot1 = plot([1:test_size],Y_test,[1:test_size],cell2mat(Y_pred)');
    %     legend('Y test','Y pred')
    %     axis([1 test_size 0.4 0.6])
    %
    %     %Plot the relative errors
    %     figure;
    %     display_str={['Autoencoder 1 Relative Mean Error = ', num2str(autoencoder_layer_1_relative_mean_error), '%'];
    %                  ['Autoencoder 2 Relative Mean Error = ', num2str(autoencoder_layer_2_relative_mean_error), '%'];
    %                  ['Autoencoder 3 Relative Mean Error = ', num2str(autoencoder_layer_3_relative_mean_error), '%'];
    %                  ['CD Prediction Relative Mean Error = ', num2str(CD_prediction_relative_mean_error), '%']};
    %     text(0.2,0.5,display_str);
    %
    cd = CD_prediction_relative_mean_error;
    
    bookKeeping(i+1,101) = CD_prediction_relative_mean_error;
    
end

