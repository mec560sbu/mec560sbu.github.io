
clc;close all;clear all

a = -1;x0 = 1; % x(0)
t_f = 2;
sys_fun = @(t,x)( a * x);
[t,x] = ode45(sys_fun,[0 t_f],x0);

figure;
plot(t,x,'rs',t,exp(a*t),'ko')
hold on;
plot(t,x,(0:0.01:t_f),exp(a*(0:0.01:t_f)),'ko')

