clear all; close all;

%% PARAMETER INITIALIZATIONS
num_samples = 500;
train_size = num_samples * 0.60;
val_size = num_samples * 0.30;
test_size = num_samples * 0.1;
sample_x_size = 32;
sample_y_size = 64;
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
     
     %% CELL2MAT
     for j = 1:train_size
         X_train_mat(:,:,1,j) =  X_train{j};
     end
     Y_train_mat = X_train_mat;
     
     for j = 1:val_size
         X_val_mat(:,:,1,j) =  X_val{j};
     end
     Y_val_mat = X_val_mat;
     
     for j = 1:test_size
         X_test_mat(:,:,1,j) =  X_test{j};
     end
     Y_test_mat = X_test_mat;
     
     
     
     %% MANUAL AUTOENCODER
     imageLayer = imageInputLayer([sample_x_size,sample_y_size,1]);
     
     encodingLayers = [ ...
        convolution2dLayer(3,16,'Padding','same'), ...
        reluLayer, ...
        maxPooling2dLayer(2,'Padding','same','Stride',2), ...
        convolution2dLayer(3,8,'Padding','same'), ...
        reluLayer, ...
        maxPooling2dLayer(2,'Padding','same','Stride',2), ...
        convolution2dLayer(3,8,'Padding','same'), ...
        reluLayer, ...
        maxPooling2dLayer(2,'Padding','same','Stride',2)];
     
     decodingLayers = [ ...
        createUpsampleTransponseConvLayer(2,8), ...
        reluLayer, ...
        createUpsampleTransponseConvLayer(2,8), ...
        reluLayer, ...
        createUpsampleTransponseConvLayer(2,16), ...
        reluLayer, ...
        convolution2dLayer(3,1,'Padding','same'), ...
        clippedReluLayer(1.0), ...
        regressionLayer];   
     
    layers = [imageLayer,encodingLayers,decodingLayers];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',100, ...
        'MiniBatchSize',15, ...
        'ValidationData',{X_val_mat,Y_val_mat}, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'Verbose',false);
    
    net = trainNetwork(X_train_mat,Y_train_mat,layers,options);
    
    Y_pred = predict(net,X_test_mat);
    %%
    figure
    subplot(1,2,1);
    %imshow(X_test_mat(:,:,:,1));
    surf(1:sample_x_size,1:sample_y_size,X_test_mat(:,:,:,1)');
    title('Real Image')
    subplot(1,2,2);
    %imshow(Y_pred(:,:,:,1));
    surf(1:sample_x_size,1:sample_y_size,Y_pred(:,:,:,1)');
    title('Reconstructed Image')
     
    %% CD CL CALCULATION FROM RECONSTRUCTED IMAGE
    if i==1
        for j=1:test_size
            decoded_field = reshape(Y_pred(:,:,:,j), sample_x_size, sample_y_size);
            for k = 1:sample_x_size
                for l = 1:sample_y_size
                    decoded_field(k,l,1) = decoded_field(k,l,1) * (max_lift_stress - min_lift_stress) + min_lift_stress;
                end
            end
            col = findPlace(g_test(j),bookKeeping_CL);
            
            fy = trapz(th,decoded_field')/(2*pi);
            
            reconstructed_CL(1,j) = bookKeeping_CL(2,col);
            reconstructed_CL(2,j) = trapz(t,fy)/(rho*U*U*R);
        end
    elseif i==2
        for j=1:test_size
            decoded_field = reshape(Y_pred(:,:,:,j), sample_x_size, sample_y_size);
            for k = 1:sample_x_size
                for l = 1:sample_y_size
                    decoded_field(k,l,1) = decoded_field(k,l,1) * (max_drag_stress - min_drag_stress) + min_drag_stress;
                end
            end
            col = findPlace(g_test(j),bookKeeping_CD);

            fx = trapz(th,decoded_field')/(2*pi);

            reconstructed_CD(1,j) = bookKeeping_CD(2,col);
            reconstructed_CD(2,j) = trapz(t,fx)/(rho*U*U*R);
        end
    end

    %%

end


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

%% CD CL error calc
CL_error = 0;
CD_error = 0;
for i = 1:test_size
    CL_error = CL_error + (reconstructed_CL(1,i) - reconstructed_CL(2,i))^2;
    CD_error = CD_error + (reconstructed_CD(1,i) - reconstructed_CD(2,i))^2;
end
CL_error = CL_error / test_size;
CD_error = CD_error / test_size;

CL_error = sqrt(CL_error);
CD_error = sqrt(CD_error);

function out = createUpsampleTransponseConvLayer(factor,numFilters)

filterSize = 2*factor - mod(factor,2); 
cropping = (factor-mod(factor,2))/2;
numChannels = 1;

out = transposedConv2dLayer(filterSize,numFilters, ... 
    'NumChannels',numChannels,'Stride',factor,'Cropping',cropping);
end

