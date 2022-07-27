clear all;

se = 0;

for i = -0.5:0.01:0.49
    se = se + i^2;
end

mse = se / 100;

rmse = sqrt(mse)

e = 0;

for i = -0.5:0.01:0.49
    e = e + abs(i);
end

me = e / 100