
clc;close all;clear all
a = -1;x0 = 1; % x(0)
t_f = 2;

figure;

for i = 1:10
    dt = i/10;
    [t,x] = solve_x(a,x0,dt,t_f)
    subplot(2,5,i)
    plot(t,x,'ro',t,exp(a*t),'ks')
    hold on
    plot(t,x,'r',(0:.01:t_f),exp(a*(0:.01:t_f)),'k')
    ylabel('x');xlabel('t')
    axis([0 2 0 1])
    title(['dt = ' num2str(dt)])
end