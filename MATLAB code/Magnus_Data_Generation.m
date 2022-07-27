clear all
clc
U = 1;
R = .1;
rho = 1;
omega = 2*pi;
% we are modeling the forces on the cylinder using the potential flow
% theory
%
%  g: is theta dependent circulation! (as if there are wholes on the
%  cylinder and they disturb the flow and change its direction).
%
vth = @(th,t,g) -2*U*R*sin(th) + (sin(omega*t)*pi + .15)*.5/pi + g*sin(th/3) ;
p   = @(th,t,g) 1/2*rho*(U^2 - vth(th,t,g).*vth(th,t,g)); 
sigmax  = @(th,t,g) cos(th).*p(th,t,g);
sigmay  = @(th,t,g) sin(th).*p(th,t,g);
Nth = 64;
Nt  = 32;
th  = [0:2*pi/(Nth-1):2*pi];
t = [0:1/(Nt-1):1]';
Ng = 500;
gg = [-1:2/(Ng-1):1];
for j=1:Ng
    g = gg(j);
    for k=1:Nt
        fx(k) = trapz(th,sigmax(th,t(k),g))/(2*pi);
        fy(k) = trapz(th,sigmay(th,t(k),g))/(2*pi);
    end
    fybar(j) = trapz(t,fy)/(rho*U*U*R);
    fxbar(j) = trapz(t,fx)/(rho*U*U*R);
end
%%
figure(1)
plot(gg,fxbar,gg,-fybar);
%grid on
%%
%figure
%contourf(sigmay(th,t,1));
%%
%figure
%contourf(sigmax(th,t,0.5))
%%
my_dir = pwd;
backslashes = strfind(my_dir,filesep);
data_dir = my_dir(1:backslashes(end)-1) + "\MATLAB data";

writematrix(fxbar, data_dir + "\Magnus_Train_Data_500s_64th_32t\CD.dat");
writematrix(-fybar, data_dir + "\Magnus_Train_Data_500s_64th_32t\CL.dat");

for i = 1:Ng
    writematrix(sigmax(th,t,gg(i)), data_dir + "\Magnus_Train_Data_500s_64th_32t\sampleCD_" + int2str(i) + ".dat");
    writematrix(-sigmay(th,t,gg(i)), data_dir + "\Magnus_Train_Data_500s_64th_32t\sampleCL_" + int2str(i) + ".dat");
end
