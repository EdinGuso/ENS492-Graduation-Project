clear all; close all;

my_dir = pwd;
backslashes = strfind(my_dir,filesep);
data_dir = my_dir(1:backslashes(end)-1) + "\MATLAB data";

data = load(data_dir + "\stresses1a.txt");
[th,is] = sort(atan2(data(:,2),data(:,1)));

%EDIT THIS
up_to_magnitude = 0.09; %availible values = 0:0.01:0.99
%EDIT THIS

last_index_for_sample = (up_to_magnitude+0.01) * 75000 + 5007;
sample_data = data(:,1:last_index_for_sample);

wait_time = 100;
wait_points = wait_time*10;
wait_columns = wait_points*5;
initial_column = wait_columns+3;

useful_data = sample_data(:,initial_column:end-5);
stresses = useful_data(:,2:5:end);
stresses_size = size(stresses);

interval = 150;

for i = 1:interval:stresses_size(2)
    tx = stresses(is,i:i+interval-1);
    
    %sum_tx = sum(tx,1);
    %[tx_peaks, tx_locs] = findpeaks(sum_tx);
    first_tx = tx(6,:);
    [tx_peaks, tx_locs] = findpeaks(first_tx);
    
    peaks_size = size(tx_locs);
    lastindex = peaks_size(2);
    start = tx_locs(lastindex-1);
    stop = tx_locs(lastindex);
    single_period_tx = tx(:,start:stop);
    
    tx_size = size(single_period_tx);
    s = tx_size(2);
    figure
    surf(th,0:s-1,single_period_tx')
end