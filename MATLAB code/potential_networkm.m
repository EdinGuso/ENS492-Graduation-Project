clear all; close all;

%% PARAMETER INITIALIZATIONS
num_samples = 500;
train_size = num_samples * 0.60;
val_size = num_samples * 0.30;
test_size = num_samples * 0.1;
sample_x_size = 25;
sample_y_size = 50;
Ng = num_samples;

bookKeeping_CL = zeros(3,Ng+1);
bookKeeping_CD = zeros(3,Ng+1);
bookKeeping_CL(3,:) = 5 * ones(1,Ng+1);
bookKeeping_CD(3,:) = 5 * ones(1,Ng+1);

reconstructed_CL = zeros(2,test_size);
reconstructed_CD = zeros(2,test_size);
th  = [0:2*pi/(sample_y_size-1):2*pi];
t = [0:1/(sample_x_size-1):1]';
U = 1;
R = .1;
rho = 1;

a1_errors = [];
a2_errors = [];
a3_errors = [];

%% LOAD DATA (X)
my_dir = pwd;
backslashes = strfind(my_dir,filesep);
data_dir = my_dir(1:backslashes(end)-1) + "\MATLAB data\Magnus_Train_Data_" + ...
    int2str(num_samples) + "s_" + num2str(sample_y_size) + "th_" + num2str(sample_x_size) + "t";
data = cell(1,num_samples);

CL_val = load(data_dir + "\CL.dat");
CD_val = load(data_dir + "\CD.dat");

bookKeeping_CL(2,1:Ng) = CL_val;
bookKeeping_CD(2,1:Ng) = CD_val;
 
CL = num2cell(CL_val(1:Ng), [1 2]);
CD = num2cell(CD_val(1:Ng), [1 2]);
for i = 1:num_samples
    sample_lift = load(data_dir + "\sampleCL_" + i + ".dat");
    sample_drag = load(data_dir + "\sampleCD_" + i + ".dat");
   
    data_lift(1,i) = num2cell(sample_lift(1:sample_x_size,1:sample_y_size), [1 2]);
    data_drag(1,i) = num2cell(sample_drag(1:sample_x_size,1:sample_y_size), [1 2]);
end

%% NORMALIZATION
max_lift_stress = intmin;
min_lift_stress = intmax;
max_drag_stress = intmin;
min_drag_stress = intmax;
for i = 1:num_samples
    if (max_lift_stress < max(data_lift{i},[],'all'))
        max_lift_stress = max(data_lift{i},[],'all');
    end
    if (min_lift_stress > min(data_lift{i},[],'all'))
        min_lift_stress = min(data_lift{i},[],'all');
    end
    if (max_drag_stress < max(data_drag{i},[],'all'))
        max_drag_stress = max(data_drag{i},[],'all');
    end
    if (min_drag_stress > min(data_drag{i},[],'all'))
        min_drag_stress = min(data_drag{i},[],'all');
    end
end

for i = 1:num_samples
    cur_lift = data_lift{i};
    cur_drag = data_drag{i};
    for j = 1:sample_x_size
        for k = 1:sample_y_size
            cur_lift(j,k) = (cur_lift(j,k) - min_lift_stress) / (max_lift_stress - min_lift_stress);
            cur_drag(j,k) = (cur_drag(j,k) - min_drag_stress) / (max_drag_stress - min_drag_stress);
        end
    end
    data_lift{i} = cur_lift;
    data_drag{i} = cur_drag;
end

%% SHUFFLE AND SPLIT
indexes = randperm(size(data, 2));

X_CL = data_lift(:,indexes);
X_CD = data_drag(:,indexes);
Y_CL = CL_val(indexes);
Y_CD = CD_val(indexes);
g = [-1:2/(Ng-1):1];

bookKeeping_CL(1,1:Ng) = g;
bookKeeping_CD(1,1:Ng) = g;

g = g(indexes);

X_CL_train = X_CL(:,1:train_size);
Y_CL_train = Y_CL(1:train_size);  
X_CD_train = X_CD(:,1:train_size);
Y_CD_train = Y_CD(1:train_size);

X_CL_val = X_CL(:,train_size+1:train_size+val_size);
Y_CL_val = Y_CL(train_size+1:train_size+val_size);
X_CD_val = X_CD(:,train_size+1:train_size+val_size);
Y_CD_val = Y_CD(train_size+1:train_size+val_size);

X_CL_test = X_CL(:,train_size+val_size+1:num_samples);
Y_CL_test = Y_CL(train_size+val_size+1:num_samples);
X_CD_test = X_CD(:,train_size+val_size+1:num_samples);
Y_CD_test = Y_CD(train_size+val_size+1:num_samples);
g_test = g(train_size+val_size+1:num_samples);
%%
 for i = 1:2
     
     if i == 1
        X_train = X_CL_train;
        X_val = X_CL_val;
        X_test = X_CL_test;

        Y_train = Y_CL_train;
        Y_val = Y_CL_val;
        Y_test = Y_CL_test;
     elseif i == 2
        X_train = X_CD_train;
        X_val = X_CD_val;
        X_test = X_CD_test;

        Y_train = Y_CD_train;
        Y_val = Y_CD_val;
        Y_test = Y_CD_test;
     end
     
    %% AUTOENCODER LAYER 1
    %Training
    hiddenSize_1 = 256;
    autoenc_1 = trainAutoencoder(X_train,hiddenSize_1,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',2,...
        'SparsityProportion',0.25,...
        'MaxEpochs', 256,...
        'ShowProgressWindow',true);
        %'EncoderTransferFunction','satlin',...
        %'DecoderTransferFunction','satlin');
    
    %Testing
    X_reconstructed = predict(autoenc_1,X_test);
    
    %Comparisson of real images and reconstructed images
    if i == 1
        figure('Name','Real Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_test{n});
            title(['C_L = ', num2str(Y_test(n))])
        end
        figure('Name','Reconstructed Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_reconstructed{n});
            title(['C_L = ', num2str(Y_test(n))])
        end
    end
    if i == 2
        figure('Name','Real Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_test{n});
            title(['C_D = ', num2str(Y_test(n))])
        end
        figure('Name','Reconstructed Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_reconstructed{n});
            title(['C_D = ', num2str(Y_test(n))])
        end
    end
    
    
    %Error calculation
    err = 0;
    max_val = intmin;
    min_val = intmax;
    for j = 1:test_size
        err = err + sum(sum((X_reconstructed{j} - X_test{j}).^2));
        if (max_val < max(max(X_test{j})))
            max_val = max(max(X_test{j}));
        end
        if (min_val > min(min(X_test{j})))
            min_val = min(min(X_test{j}));
        end
    end
    merr = err / test_size;
    mmerr = sqrt(merr / (sample_x_size*sample_y_size));
    autoencoder_layer_1_relative_mean_error = 100 * mmerr;
    %autoencoder_layer_1_relative_mean_error = 100 * (mmerr - min_val) / (max_val - min_val);
    
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
        'ShowProgressWindow',true);
        %'EncoderTransferFunction','satlin',...
        %'DecoderTransferFunction','satlin');
    
    %Testing
    X_encoded_reconstructed = predict(autoenc_2,X_test_encoded);
    X_reconstructed_reconstructed = decode(autoenc_1,cell2mat(X_encoded_reconstructed));
    
    %Comparisson of real images and reconstructed images
    if i == 1
        figure('Name','Real Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_test{n});
            title(['C_L = ', num2str(Y_test(n))])
        end
        figure('Name','Reconstructed Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_reconstructed_reconstructed{n});
            title(['C_L = ', num2str(Y_test(n))])
        end
    end
    if i == 2
        figure('Name','Real Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_test{n});
            title(['C_D = ', num2str(Y_test(n))])
        end
        figure('Name','Reconstructed Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_reconstructed_reconstructed{n});
            title(['C_D = ', num2str(Y_test(n))])
        end
    end
    
    
    %Error calculation
    err = 0;
    max_val = intmin;
    min_val = intmax;
    for k = 1:test_size
        err = err + sum(sum((X_reconstructed_reconstructed{k} - X_test{k}).^2));
        if (max_val < max(max(X_test{k})))
            max_val = max(max(X_test{k}));
        end
        if (min_val > min(min(X_test{k})))
            min_val = min(min(X_test{k}));
        end
    end
    merr = err / test_size;
    mmerr = sqrt(merr / (sample_x_size*sample_y_size));
    autoencoder_layer_2_relative_mean_error = 100 * mmerr;
    %autoencoder_layer_2_relative_mean_error = 100 * (mmerr - min_val) / (max_val - min_val);
    
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
        'ShowProgressWindow',true);
        %'EncoderTransferFunction','satlin',...
        %'DecoderTransferFunction','satlin');
    
    %Testing
    X_encoded_encoded_reconstructed = predict(autoenc_3,X_test_encoded_encoded);
    X_reconstructed_reconstructed_reconstructed = decode(autoenc_1,cell2mat(decode(autoenc_2,cell2mat(X_encoded_encoded_reconstructed))));
    
        %Comparisson of real images and reconstructed images
    if i == 1
        figure('Name','Real Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_test{n});
            title(['C_L = ', num2str(Y_test(n))])
        end
        figure('Name','Reconstructed Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_reconstructed_reconstructed_reconstructed{n});
            title(['C_L = ', num2str(Y_test(n))])
        end
    end
    if i == 2
        figure('Name','Real Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_test{n});
            title(['C_D = ', num2str(Y_test(n))])
        end
        figure('Name','Reconstructed Test Data');
        for n = 1:10
            subplot(2,10/2,n);
            imshow(X_reconstructed_reconstructed_reconstructed{n});
            title(['C_D = ', num2str(Y_test(n))])
        end
    end
    
    
    %Error calculation
    err = 0;
    max_val = intmin;
    min_val = intmax;
    for m = 1:test_size
        err = err + sum(sum((X_reconstructed_reconstructed_reconstructed{m} - X_test{m}).^2));
        if (max_val < max(max(X_test{m})))
            max_val = max(max(X_test{m}));
        end
        if (min_val > min(min(X_test{m})))
            min_val = min(min(X_test{m}));
        end
    end
    merr = err / test_size;
    mmerr = sqrt(merr / (sample_x_size*sample_y_size));
    autoencoder_layer_3_relative_mean_error = 100 * mmerr;
    %autoencoder_layer_3_relative_mean_error = 100 * (mmerr - min_val) / (max_val - min_val);
    
    %% CD CL CALCULATION FROM RECONSTRUCTED IMAGE
    if i==1
        for j=1:test_size
            decoded_field = X_reconstructed_reconstructed_reconstructed{j};
            for k = 1:sample_x_size
                for l = 1:sample_y_size
                    decoded_field(k,l) = decoded_field(k,l) * (max_lift_stress - min_lift_stress) + min_lift_stress;
                end
            end
            col = findPlace(g_test(j),bookKeeping_CL);
            
            fy = trapz(th,decoded_field')/(2*pi);
            
            reconstructed_CL(1,j) = bookKeeping_CL(2,col);
            reconstructed_CL(2,j) = trapz(t,fy)/(rho*U*U*R);
        end
    elseif i==2
        for j=1:test_size
            decoded_field = X_reconstructed_reconstructed_reconstructed{j};
            for k = 1:sample_x_size
                for l = 1:sample_y_size
                    decoded_field(k,l) = decoded_field(k,l) * (max_drag_stress - min_drag_stress) + min_drag_stress;
                end
            end
            col = findPlace(g_test(j),bookKeeping_CD);

            fx = trapz(th,decoded_field')/(2*pi);

            reconstructed_CD(1,j) = bookKeeping_CD(2,col);
            reconstructed_CD(2,j) = trapz(t,fx)/(rho*U*U*R);
        end
    end
    
    
    %% PCA
    %asda = cell2mat(X_train_encoded_encoded_encoded)';
    %coef = pca(asda)


    %% SHALLOW NETWORK
    %Encode the data
    X_train_encoded_encoded_encoded = num2cell(encode(autoenc_3, X_train_encoded_encoded), 1);
    X_val_encoded_encoded_encoded = num2cell(encode(autoenc_3, X_val_encoded_encoded), 1);
    X_test_encoded_encoded_encoded = num2cell(encode(autoenc_3, X_test_encoded_encoded), 1);
    
    %Define the NN architecture
    layers = [
        sequenceInputLayer(hiddenSize_3,'Name','Encoded Data Input Layer')
        fullyConnectedLayer(64,'Name','Fully Connected Regression Layer-1')
        tanhLayer('Name','Activation Function Layer-1')
        fullyConnectedLayer(16,'Name','Fully Connected Regression Layer-2')
        tanhLayer('Name','Activation Function Layer-2')
        fullyConnectedLayer(1,'Name','Fully Connected Regression Layer-3')
        tanhLayer('Name','Activation Function Layer-3')
        regressionLayer('Name','Output Layer')];
    
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',3, ...
        'MaxEpochs',15, ...
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',10, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{X_val_encoded_encoded_encoded,num2cell(Y_val)}, ...
        'ValidationFrequency',1, ...
        'Plots','training-progress', ...
        'Verbose',false);
        %'Plots','none', ...
    %Train and test
    net = trainNetwork(X_train_encoded_encoded_encoded,num2cell(Y_train),layers,options);
    Y_pred = predict(net,X_test_encoded_encoded_encoded);
    
    comp=[Y_test', cell2mat(Y_pred)];

    %% BOOKKEEPING
    %Error calculation
    err = 0;
    max_val = intmin;
    min_val = intmax;
    for n = 1:test_size
        err = err + (Y_pred{n} - Y_test(n))^2; 
        if(i == 1) % if it is CL
            col = findPlace(g_test(n),bookKeeping_CL);
            bookKeeping_CL(3,col) = Y_pred{n};
        else
            col = findPlace(g_test(n),bookKeeping_CD);
            bookKeeping_CD(3,col) = Y_pred{n};
        end
        if (max_val < max(Y_test(n)))
            max_val = max(Y_test(n));
        end
        if (min_val > min(Y_test(n)))
            min_val = min(Y_test(n));
        end
    end
    val_range = max_val - min_val;
    
    merr = sqrt(err / test_size);
    %%
    if(i == 1)
        CL_prediction_relative_mean_error = 100 * merr;
        bookKeeping_CL(3,Ng+1) = CL_prediction_relative_mean_error;
    else
        CD_prediction_relative_mean_error = 100 * merr;
        bookKeeping_CD(3,Ng+1) = CD_prediction_relative_mean_error;
    end
    a1_errors(i) = autoencoder_layer_1_relative_mean_error;
    a2_errors(i) = autoencoder_layer_2_relative_mean_error;
    a3_errors(i) = autoencoder_layer_3_relative_mean_error;
end

%% PLOTTING
figure
plot(bookKeeping_CL(1,1:num_samples),bookKeeping_CL(2,1:num_samples),'-b')
hold on
plot(bookKeeping_CD(1,1:num_samples),bookKeeping_CD(2,1:num_samples),'-r')

yplot= bookKeeping_CL(3,1:num_samples);      % make a copy of the data specifically for plotting
yplot(yplot==0)=nan;                         % replace 0 elements with NaN
plot(bookKeeping_CL(1,1:num_samples),yplot,'+b')

yplot= bookKeeping_CD(3,1:num_samples);      % make a copy of the data specifically for plotting
yplot(yplot==0)=nan;                         % replace 0 elements with NaN
plot(bookKeeping_CD(1,1:num_samples),yplot,'+r')

ylim([-2,2])
xticks(-1:0.2:1)
xlabel('g')
ylabel('C_D and C_L')
legend('Real C_L','Real C_D','Predicted C_L','Predicted C_D')

hold off

figure
plot(g_test,reconstructed_CL(1,:),'+b')
hold on
plot(g_test,reconstructed_CL(2,:),'+r')
hold off
xticks(-1:0.2:1)
xlabel('g')
ylabel('C_L')
legend('Real C_L','Reconstructed C_L')

figure
plot(g_test,reconstructed_CD(1,:),'+b')
hold on
plot(g_test,reconstructed_CD(2,:),'+r')
hold off
xticks(-1:0.2:1)
xlabel('g')
ylabel('C_D')
legend('Real C_D','Reconstructed C_D')

%% STATS

errors_CL =bookKeeping_CL(3,Ng+1) ;
%stdE_CL = std(errors_CL);
%meanE_CL = mean(errors_CL);

errors_CD =bookKeeping_CD(3,Ng+1) ;
%stdE_CD = std(errors_CD);
%meanE_CD = mean(errors_CD);

% std_a1 = std(a1_errors);
% mean_a1 = mean(a1_errors);
% 
% std_a2 = std(a2_errors);
% mean_a2 = mean(a2_errors);
% 
% std_a3 = std(a3_errors);
% mean_a3 = mean(a3_errors);

